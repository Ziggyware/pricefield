// deno task test   (KV_PATH=:memory: keeps tests off the real database)
import { assert, assertAlmostEquals } from "@std/assert";
import { HORIZONS, PAPER } from "./config.ts";
import { newAssetState, step, summary } from "./engine.ts";
import type { Bar } from "./feeds.ts";
import { learn, logit, newLearner, predictive, prob } from "./model.ts";
import { newPaper, stepPaper } from "./paper.ts";
import { dieboldMariano, metrics } from "./score.ts";
import { hurst, kalmanSlope, mean, normCdf, normInv } from "./stats.ts";

/** Deterministic Gaussian stream (xorshift + Box–Muller). */
function rng(seed: number) {
  let s = seed >>> 0 || 1;
  const u = () => ((s ^= s << 13), (s ^= s >>> 17), (s ^= s << 5), (s >>> 0) / 4294967296 + 1e-12);
  return () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
}

/** 1-minute bars from a return process; φ = AR(1) coefficient on 1-minute log returns. */
function bars(n: number, phi: number, seed: number): Bar[] {
  const g = rng(seed), out: Bar[] = [];
  let p = 50_000, r = 0;
  for (let i = 0; i < n; i++) {
    r = phi * r + 0.0008 * g();
    const o = p;
    p *= Math.exp(r);
    out.push({
      t: 1_700_000_000 + i * 60,
      o,
      h: Math.max(o, p) * (1 + 0.0002 * Math.abs(g())),
      l: Math.min(o, p) * (1 - 0.0002 * Math.abs(g())),
      c: p,
      v: 1 + Math.abs(g()),
    });
  }
  return out;
}

function replay(b: Bar[]) {
  const S = newAssetState("BTC");
  for (let i = 0; i < b.length; i++) step(S, b, i, {}, 0, false);
  return S;
}

Deno.test("normInv inverts normCdf", () => {
  for (const p of [1e-6, 0.025, 0.16, 0.5, 0.84, 0.975, 1 - 1e-6]) assertAlmostEquals(normCdf(normInv(p)), p, 2e-7);
});

Deno.test("Anis–Lloyd Hurst: ~0.5 for white noise, > 0.6 for persistent increments", () => {
  const hs = Array.from({ length: 40 }, (_, k) => {
    const g = rng(k + 1);
    return hurst(Array.from({ length: 64 }, g));
  });
  assertAlmostEquals(mean(hs), 0.5, 0.06);
  const g = rng(99);
  let a = 0;
  const persistent = Array.from({ length: 64 }, () => (a = 0.8 * a + g()));
  assert(hurst(persistent) > 0.6);
});

Deno.test("Kalman slope recovers a linear drift", () => {
  const g = rng(7), drift = 2e-4, logp = Array.from({ length: 240 }, (_, i) => i * drift + 1e-3 * g());
  assertAlmostEquals(kalmanSlope(logp, 1e-3), drift, 1e-4);
});

Deno.test("online logistic learns the sign and calibration of a known model", () => {
  const g = rng(3), m = newLearner(), truth = [0.2, 1.5, -1];
  let ll = 0, n = 0;
  for (let k = 0; k < 20000; k++) {
    const x = [1, g(), g(), ...new Array(13).fill(0)], z = truth[0] + truth[1] * x[1] + truth[2] * x[2];
    const y = Math.random() < 1 / (1 + Math.exp(-z)) ? 1 : 0;
    if (k > 15000) {
      const p = prob(m, x);
      ll += -Math.log(y ? p : 1 - p);
      n++;
    }
    learn(m, x, logit(m, x), y as 0 | 1);
  }
  assert(m.w[1] > 1 && m.w[2] < -0.6, `weights ${m.w.slice(0, 3)}`);
  assert(ll / n < 0.58, `held-out log loss ${ll / n}`); // Bayes rate for this model is ≈ 0.55
});

Deno.test("FHS predictive distribution puts exactly p of its mass above zero", () => {
  const g = rng(11), zres = Array.from({ length: 500 }, () => g() * (1 + 0.5 * Math.abs(g())));
  for (const p of [0.3, 0.5, 0.62]) {
    const { q } = predictive(zres, p);
    assert(q[0] < q[1] && q[1] < q[2] && q[2] < q[3] && q[3] < q[4]);
    const med = [...zres].sort((a, b) => a - b)[250], m = q[2];
    const above = zres.filter((z) => z - med + m > 0).length / zres.length;
    assertAlmostEquals(above, p, 0.01);
  }
});

Deno.test("Diebold–Mariano: zero for identical losses, significant for a dominated forecaster", () => {
  assertAlmostEquals(dieboldMariano(new Array(500).fill(0.1), 4).stat, 0);
  const g = rng(5);
  assert(dieboldMariano(Array.from({ length: 500 }, () => 0.05 + 0.1 * g()), 4).stat > 3);
});

Deno.test("prequential engine: no skill on a random walk, real skill on AR(1) returns", () => {
  const rw = summary(replay(bars(4000, 0, 21))), ar = summary(replay(bars(8000, 0.4, 21)));
  const r5 = rw.horizons[5].metrics, a5 = ar.horizons[5].metrics;
  assert(r5.n === 1000 && a5.n === 1000); // SCORE_N most recent matured forecasts
  assert(r5.vsClim.stat < 1.96, `random walk must not beat climatology: DM ${r5.vsClim.stat}`);
  assert(a5.bss > 0.01 && a5.vsClim.stat > 1.96, `AR(1) skill: BSS ${a5.bss}, DM ${a5.vsClim.stat}`);
  assert(ar.horizons[5].champion.w.mom1 > 0.2, "momentum weight must be learned positive under positive autocorrelation");
  // Band coverage is testable only where the 1000 scored forecasts hold many independent outcomes (5 m ⇒ ~200).
  const c5 = rw.horizons[5].metrics;
  for (const h of HORIZONS) {
    console.log(`  RW ${h}m coverage ${rw.horizons[h].metrics.cov68.toFixed(3)} / ${rw.horizons[h].metrics.cov95.toFixed(3)}`);
  }
  assertAlmostEquals(c5.cov68, 0.68, 0.06);
  assertAlmostEquals(c5.cov95, 0.95, 0.03);
  console.log(`  RW 5m  BSS ${r5.bss.toFixed(4)} DM ${r5.vsClim.stat.toFixed(2)} cov ${r5.cov68.toFixed(3)}/${r5.cov95.toFixed(3)}`);
  console.log(
    `  AR 5m  BSS ${a5.bss.toFixed(4)} DM ${a5.vsClim.stat.toFixed(2)} hit ${
      a5.hit.toFixed(3)
    } · paper ${ar.paper.stats.n} trades, equity ${ar.paper.equity.toFixed(0)}`,
  );
});

Deno.test("paper: a flat round trip loses exactly maker + taker fees", () => {
  const pp = newPaper(), bar = (t: number, l: number, h: number, c: number): Bar => ({ t, o: c, h, l, c, v: 1 });
  stepPaper(pp, bar(0, 100, 100, 100), {
    p: 0.9,
    er: 0.02,
    sd: 0.02,
    tp: 110,
    sl: 90,
    tpShort: 90,
    slShort: 110,
    touch: { bid: 100, ask: 100.1 },
    eventRisk: 0,
  }, false);
  const qty = pp.order!.qty;
  stepPaper(pp, bar(60, 99.9, 100, 100), null, false); // trades through the bid: maker fill at 100
  for (let t = 120; t <= 60 * PAPER.horizon; t += 60) stepPaper(pp, bar(t, 99.95, 100.05, 100), null, false);
  assert(pp.trades.length === 1 && pp.trades[0].reason === "horizon");
  assertAlmostEquals(pp.equity, PAPER.startEquity - qty * 100 * (PAPER.makerFee + PAPER.takerFee), 1e-6);
});

Deno.test("metrics on an empty buffer are well-defined", () => {
  const m = metrics(new Float32Array(0), 5);
  assert(m.n === 0 && m.vsClim.p === 1);
});
