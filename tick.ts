import { type Asset, ASSETS } from "./config.ts";
import { newAssetState, step } from "./engine.ts";
import { type Bar, coinbaseBars, snapshot } from "./feeds.ts";
import { journal, kv, loadAsset, saveAsset, withLock } from "./store.ts";
import { contextFor, getContext } from "./swarm.ts";

/** Coinbase 1-minute history in 300-bar pages, ascending, completed bars only. */
async function history(a: Asset, nowSec: number, pages: number): Promise<Bar[]> {
  const end = Math.floor(nowSec / 60) * 60, seen = new Map<number, Bar>();
  for (let k = pages - 1; k >= 0; k--) {
    const hi = end - k * 300 * 60, lo = hi - 300 * 60;
    for (const b of await coinbaseBars(a, lo, hi)) if (b.t + 60 <= nowSec) seen.set(b.t, b);
  }
  return [...seen.values()].sort((x, y) => x.t - y.t);
}

export interface Health {
  t: number;
  ms: number;
  assets: Record<string, { bars: number; replayed: number; errors: string[]; lastBar: number }>;
}

/**
 * Minute cron. First run per asset replays ~30 h of Coinbase history prequentially (every forecast is scored before
 * its outcome is learned), so the scoreboard is honest from the start. Later runs advance every new completed bar;
 * only the newest bar sees live books, trades, positioning and agent context.
 */
export async function tick(nowSec = Math.floor(Date.now() / 1000)): Promise<Health> {
  const t0 = performance.now(), ctx = await getContext();
  const health: Health = { t: nowSec, ms: 0, assets: {} };
  await Promise.all(ASSETS.map((a) =>
    withLock(a, async () => {
      const snap = await snapshot(a, nowSec);
      const gmClose = new Map(snap.gm.map((b) => [b.t, b.c]));
      const c = contextFor(ctx, a, nowSec);
      let S = await loadAsset(a), replayed = 0;
      if (!S) {
        S = newAssetState(a);
        const hist = await history(a, nowSec, 6);
        for (let i = 0; i < hist.length; i++, replayed++) step(S, hist, i, { gmClose }, 0, false);
        await journal({ t: nowSec, agent: "kernel", asset: a, text: `replayed ${hist.length} bars of history prequentially` });
      }
      const bars = snap.cb, fresh = bars.map((b, i) => [b, i] as const).filter(([b]) => b.t > S!.lastBar);
      for (const [, i] of fresh) {
        const newest = i === bars.length - 1;
        step(
          S,
          bars,
          i,
          newest
            ? {
              gmClose,
              books: [snap.cbBook, snap.gmBook],
              trades: snap.trades,
              funding: snap.funding,
              oi: snap.oi,
              sentiment: c.sentiment,
            }
            : { gmClose },
          newest ? c.eventRisk : 0,
          true,
        );
      }
      const last = bars.at(-1);
      if (fresh.length && last && S.latest && Math.abs(Math.log(last.c / bars.at(-2)!.c) / S.latest.sigma) >= 4) {
        await kv.set(["anomaly", a], { t: last.t, z: Math.log(last.c / bars.at(-2)!.c) / S.latest.sigma, price: last.c }, {
          expireIn: 3_600_000,
        });
      }
      for (const e of S.events) await journal({ t: nowSec, agent: "kernel", asset: a, text: e });
      await saveAsset(S);
      health.assets[a] = { bars: fresh.length, replayed, errors: snap.errors, lastBar: S.lastBar };
    }).catch((e) => {
      health.assets[a] = { bars: 0, replayed: 0, errors: [String(e?.message ?? e)], lastBar: 0 };
    })
  ));
  health.ms = Math.round(performance.now() - t0);
  await kv.set(["health"], health);
  return health;
}
