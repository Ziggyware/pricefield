import { ASSETS, FEATURES, HORIZONS, PAPER, ROUTER_KEY } from "./config.ts";
import { summary } from "./engine.ts";
import { getJournal, journal, kv, loadAsset } from "./store.ts";
import { analystAgent, contextAgent, criticAgent, getContext } from "./swarm.ts";
import { tick } from "./tick.ts";

/** Agent failures are journaled, never thrown into the scheduler. */
const guarded = (agent: string, fn: (nowSec: number) => Promise<void>) => async () => {
  const now = Math.floor(Date.now() / 1000);
  try {
    await fn(now);
  } catch (e) {
    await journal({ t: now, agent, text: `failed: ${(e as Error).message}` });
  }
};

// Crons must be registered at top level, before Deno.serve (Deno Deploy discovers them at deploy time). All UTC.
Deno.cron("pricefield tick", "* * * * *", async () => void (await tick()));
Deno.cron("pricefield context agent", "*/15 * * * *", guarded("context", contextAgent));
Deno.cron("pricefield analyst agent", "*/5 * * * *", guarded("analyst", analystAgent));
Deno.cron("pricefield critic agent", "7 * * * *", guarded("critic", criticAgent));

const ui = (name: string) => Deno.readTextFile(new URL(`./ui/${name}`, import.meta.url));
const types: Record<string, string> = { "index.html": "text/html", "app.js": "text/javascript", "app.css": "text/css" };
const json = (v: unknown, status = 200) => new Response(JSON.stringify(v), { status, headers: { "content-type": "application/json" } });
const ADMIN = Deno.env.get("ADMIN_TOKEN") ?? "";

Deno.serve(async (req) => {
  const { pathname } = new URL(req.url);
  if (req.method === "GET" && (pathname === "/" || pathname.slice(1) in types)) {
    const name = pathname === "/" ? "index.html" : pathname.slice(1);
    return new Response(await ui(name), { headers: { "content-type": `${types[name]}; charset=utf-8`, "cache-control": "no-cache" } });
  }
  if (req.method === "GET" && pathname === "/api/state") {
    const states = await Promise.all(ASSETS.map(loadAsset));
    return json({
      assets: states.filter((s) => s !== null).map(summary),
      context: await getContext(),
      journal: (await getJournal()).slice(-60).reverse(),
      health: (await kv.get(["health"])).value,
      config: { horizons: HORIZONS, features: FEATURES, paper: PAPER, router: !!ROUTER_KEY },
    });
  }
  // Manual triggers, enabled only when ADMIN_TOKEN is set and supplied.
  if (req.method === "POST" && pathname.startsWith("/api/run/")) {
    if (!ADMIN || req.headers.get("x-admin-token") !== ADMIN) return json({ error: "forbidden" }, 403);
    const name = pathname.slice(9);
    const run: Record<string, () => Promise<unknown>> = {
      tick: () => tick(),
      context: guarded("context", contextAgent),
      analyst: guarded("analyst", analystAgent),
      critic: guarded("critic", criticAgent),
    };
    if (!run[name]) return json({ error: "unknown" }, 404);
    return json({ ok: true, result: await run[name]() ?? null });
  }
  return json({ error: "not found" }, 404);
});
