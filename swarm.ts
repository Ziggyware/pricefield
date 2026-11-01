import { type Asset, ASSETS, FEATURES, HORIZONS, HYPER_BOUNDS, NEWS_FEEDS, ROUTER_KEY, ROUTER_URL } from "./config.ts";
import { summary } from "./engine.ts";
import { newLearner } from "./model.ts";
import { journal, kv, loadAsset, saveAsset, withLock } from "./store.ts";
import { clamp } from "./stats.ts";

/**
 * nokey-router, OpenAI chat-completions schema. Agents never see the key; this client injects it.
 * An alias the router rejects (4xx other than auth) falls back to free:smart, the alias its own UI uses.
 */
async function chat(alias: string, system: string, user: string): Promise<{ text: string; model: string }> {
  if (!ROUTER_KEY) throw new Error("ROUTER_API_KEY is not set");
  for (const model of alias === "free:smart" ? [alias] : [alias, "free:smart"]) {
    const r = await fetch(`${ROUTER_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${ROUTER_KEY}` },
      body: JSON.stringify({ model, temperature: 0, messages: [{ role: "system", content: system }, { role: "user", content: user }] }),
      signal: AbortSignal.timeout(90_000),
    });
    const j = await r.json().catch(() => ({}));
    if (r.ok) return { text: j.choices?.[0]?.message?.content ?? "", model: j.model ?? model };
    if (r.status === 401 || r.status === 403 || r.status >= 500 || model === "free:smart") {
      throw new Error(`router ${r.status} (${model}): ${j?.error?.message ?? ""}`);
    }
  }
  throw new Error("unreachable");
}

/** First balanced JSON object in a model reply. */
function json<T>(s: string): T {
  const a = s.indexOf("{"), b = s.lastIndexOf("}");
  if (a < 0 || b <= a) throw new Error("no JSON object in reply");
  return JSON.parse(s.slice(a, b + 1)) as T;
}

export interface Context {
  t: number;
  model: string;
  regime: string;
  note: string;
  assets: Record<Asset, { sentiment: number; eventRisk: number }>;
}

export async function getContext(): Promise<Context | null> {
  return (await kv.get<Context>(["context"])).value;
}

/** Sentiment decays with a two-hour half-life; stale context contributes nothing. */
export function contextFor(c: Context | null, a: Asset, nowSec: number) {
  if (!c) return { sentiment: 0, eventRisk: 0 };
  const age = nowSec - c.t, w = Math.pow(2, -age / 7200);
  return { sentiment: c.assets[a].sentiment * w, eventRisk: age < 3600 ? c.assets[a].eventRisk : 0 };
}

async function headlines(): Promise<string[]> {
  const out: { t: number; s: string }[] = [];
  await Promise.all(NEWS_FEEDS.map(async (u) => {
    const xml = await fetch(u, { signal: AbortSignal.timeout(10_000) }).then((r) => r.text()).catch(() => "");
    for (
      const m of xml.matchAll(/<item>[\s\S]*?<title>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/title>[\s\S]*?<pubDate>([\s\S]*?)<\/pubDate>/g)
    ) {
      const t = Date.parse(m[2]);
      if (Number.isFinite(t)) out.push({ t, s: `${new Date(t).toISOString().slice(0, 16)}Z ${m[1].trim()}` });
    }
  }));
  return out.sort((a, b) => b.t - a.t).slice(0, 30).map((x) => x.s);
}

/** Context agent: news + positioning → per-asset sentiment and event risk, which enter the model as a learned feature. */
export async function contextAgent(nowSec: number): Promise<void> {
  const news = await headlines();
  const market = await Promise.all(ASSETS.map(async (a) => {
    const S = await loadAsset(a);
    return S?.latest
      ? `${a} ${S.latest.price.toFixed(2)} σ1m ${(S.latest.sigma * 1e4).toFixed(1)}bp funding_z ${
        S.latest.x[FEATURES.indexOf("funding_z")].toFixed(2)
      }`
      : `${a} warming up`;
  }));
  const sys = "You are a crypto market context analyst. Judge only from the headlines and data given. Return only JSON.";
  const user =
    `Now ${new Date(nowSec * 1e3).toISOString()}.\nMarket:\n${market.join("\n")}\nHeadlines (newest first):\n${news.join("\n")}\n\n` +
    `Return {"BTC":{"sentiment":s,"eventRisk":e},"ETH":{"sentiment":s,"eventRisk":e},"regime":"≤6 words","note":"≤60 words"}. ` +
    `sentiment in [-1,1]: net directional implication for the next hours (0 when headlines are irrelevant or mixed). ` +
    `eventRisk in [0,1]: probability-like weight that a scheduled or breaking event drives an outsized move in the next hour.`;
  const r = await chat("free:smart", sys, user);
  const j = json<Record<string, { sentiment: number; eventRisk: number }> & { regime: string; note: string }>(r.text);
  const assets = Object.fromEntries(ASSETS.map((a) => [a, {
    sentiment: clamp(+j[a]?.sentiment || 0, -1, 1),
    eventRisk: clamp(+j[a]?.eventRisk || 0, 0, 1),
  }])) as Context["assets"];
  const c: Context = {
    t: nowSec,
    model: r.model,
    regime: String(j.regime ?? "").slice(0, 60),
    note: String(j.note ?? "").slice(0, 500),
    assets,
  };
  await kv.set(["context"], c);
  await journal({
    t: nowSec,
    agent: "context",
    model: r.model,
    text: `${c.regime}: ${
      ASSETS.map((a) => `${a} s ${assets[a].sentiment.toFixed(2)} risk ${assets[a].eventRisk.toFixed(2)}`).join(" · ")
    }. ${c.note}`,
  });
}

/** Analyst agent: explains flagged ≥ 4σ one-minute moves using the latest context. */
export async function analystAgent(nowSec: number): Promise<void> {
  for (const a of ASSETS) {
    const flag = await kv.get<{ t: number; z: number; price: number }>(["anomaly", a]);
    if (!flag.value) continue;
    await kv.delete(["anomaly", a]);
    const c = await getContext(), news = await headlines();
    const r = await chat(
      "free:fast",
      "You explain sudden crypto price moves concisely and say when the cause is unknown. Plain text, ≤ 80 words.",
      `${a} moved ${flag.value.z.toFixed(1)}σ in one minute to ${flag.value.price} at ${new Date(flag.value.t * 1e3).toISOString()}.\n` +
        `Context: ${c ? `${c.regime}. ${c.note}` : "none"}\nHeadlines:\n${news.slice(0, 15).join("\n")}`,
    );
    await journal({ t: nowSec, agent: "analyst", asset: a, model: r.model, text: r.text.trim().slice(0, 800) });
  }
}

/**
 * Critic agent: reads the scoreboard and proposes at most two challengers (bounded hyper-parameters and ablations).
 * The kernel decides promotion by a Diebold–Mariano test on live log loss; the critic cannot change a champion.
 */
export async function criticAgent(nowSec: number): Promise<void> {
  const states = (await Promise.all(ASSETS.map(loadAsset))).filter((s) => s !== null);
  const board = states.map((S) => {
    const s = summary(S);
    return {
      asset: S.asset,
      horizons: Object.fromEntries(HORIZONS.map((h) => {
        const v = s.horizons[h], m = v.metrics;
        return [h, {
          n: m.n,
          bss: +m.bss.toFixed(4),
          logLoss: +m.logLoss.toFixed(4),
          logLossClim: +m.logLossClim.toFixed(4),
          dmVsClim: +m.vsClim.stat.toFixed(2),
          cov68: +m.cov68.toFixed(3),
          cov95: +m.cov95.toFixed(3),
          champion: { eta: v.champion.eta, l2: v.champion.l2, off: v.champion.off, w: v.champion.w },
          challengerActive: !!v.challenger,
        }];
      })),
      paper: { equity: Math.round(s.paper.equity), ...s.paper.stats },
    };
  });
  const sys = "You tune an online logistic forecaster. Propose only changes a paired live test can confirm. Return only JSON.";
  const user = `Scoreboard:\n${JSON.stringify(board)}\nFeatures: ${FEATURES.join(", ")}\n` +
    `Return {"proposals":[{"asset":"BTC|ETH","horizon":5|15|60,"eta":${HYPER_BOUNDS.eta.join("..")},"l2":${
      HYPER_BOUNDS.l2.join("..")
    },"off":["feature",...],"why":"≤30 words"}],"assessment":"≤60 words"}. ` +
    `At most 2 proposals, only for horizons without an active challenger; [] when nothing is justified.`;
  const r = await chat("free:reasoning", sys, user);
  const j = json<
    { proposals?: { asset: Asset; horizon: number; eta: number; l2: number; off: string[]; why: string }[]; assessment?: string }
  >(r.text);
  const made: string[] = [];
  for (const p of (j.proposals ?? []).slice(0, 2)) {
    const h = HORIZONS.find((x) => x === +p.horizon);
    if (!ASSETS.includes(p.asset) || !h) continue;
    // Re-read under the asset lock: the tick cron may have advanced the state since the scoreboard was built.
    const line = await withLock(p.asset, async () => {
      const S = await loadAsset(p.asset);
      if (!S || S.models[h].chal) return null;
      const champ = S.models[h].champ, chal = newLearner({ eta: +p.eta, l2: +p.l2, off: p.off ?? [] });
      chal.w = [...champ.w];
      chal.g2 = [...champ.g2];
      Object.assign(S.models[h], { chal, chalSince: S.lastBar, chalWhy: String(p.why ?? "").slice(0, 200) });
      await saveAsset(S);
      return `${p.asset} ${h}m η ${chal.eta} λ ${chal.l2} off [${chal.off.join(",")}]`;
    });
    if (line) made.push(line);
  }
  await journal({
    t: nowSec,
    agent: "critic",
    model: r.model,
    text: `${String(j.assessment ?? "").slice(0, 500)}${made.length ? ` Challengers: ${made.join("; ")}.` : " No challenger installed."}`,
  });
}
