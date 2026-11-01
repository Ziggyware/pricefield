import { SCORE_N } from "./config.ts";
import { logLoss } from "./model.ts";
import { normCdf } from "./stats.ts";

/** One matured forecast: [t, p, p_challenger (NaN if none), p_climatology, y, z, in68, in95]. */
export const REC = 8;

export function pushRecord(buf: Float32Array, rec: number[]): Float32Array {
  const n = Math.min(buf.length / REC + 1, SCORE_N), out = new Float32Array(n * REC);
  out.set(buf.subarray(buf.length - (n - 1) * REC), 0);
  out.set(rec, (n - 1) * REC);
  return out;
}

export function rows(buf: Float32Array): number[][] {
  const r: number[][] = [];
  for (let i = 0; i < buf.length; i += REC) r.push(Array.from(buf.subarray(i, i + REC)));
  return r;
}

/**
 * Diebold–Mariano test on loss differentials d = loss_a − loss_b with a Newey–West (Bartlett) long-run variance.
 * Overlapping h-minute forecasts sampled every minute are MA(h−1) under the null, so lag = h − 1.
 * stat > 0 ⇒ b has lower loss. p is two-sided.
 */
export function dieboldMariano(d: number[], lag: number): { stat: number; p: number; n: number } {
  const n = d.length;
  if (n < 30) return { stat: 0, p: 1, n };
  const m = d.reduce((s, x) => s + x, 0) / n;
  const gamma = (k: number) => {
    let s = 0;
    for (let t = k; t < n; t++) s += (d[t] - m) * (d[t - k] - m);
    return s / n;
  };
  let lrv = gamma(0);
  const L = Math.min(lag, n - 1);
  for (let k = 1; k <= L; k++) lrv += 2 * (1 - k / (L + 1)) * gamma(k);
  // A (numerically) constant differential carries no evidence either way.
  if (!(lrv > 1e-14 * (1 + m * m))) return { stat: 0, p: 1, n };
  const stat = m / Math.sqrt(lrv / n);
  return { stat, p: 2 * (1 - normCdf(Math.abs(stat))), n };
}

export interface Metrics {
  n: number;
  brier: number;
  brierClim: number;
  /** Brier skill score vs climatology: 1 − B/B_clim. > 0 means better than the base rate. */
  bss: number;
  logLoss: number;
  logLossClim: number;
  /** Fraction of outcomes on the predicted side of 0.5. */
  hit: number;
  /** Empirical coverage of the nominal 68 % and 95 % bands. */
  cov68: number;
  cov95: number;
  /** DM on log loss, climatology − model (stat > 0 ⇒ model better). */
  vsClim: { stat: number; p: number; n: number };
  /** DM on log loss, champion − challenger (stat > 0 ⇒ challenger better), paired records only. */
  vsChallenger: { stat: number; p: number; n: number };
  /** Reliability diagram: ten probability bins. */
  bins: { p: number; freq: number; n: number }[];
}

export function metrics(buf: Float32Array, horizon: number): Metrics {
  const r = rows(buf), n = r.length;
  const avg = (f: (x: number[]) => number) => (n ? r.reduce((s, x) => s + f(x), 0) / n : NaN);
  const brier = avg((x) => (x[1] - x[4]) ** 2), brierClim = avg((x) => (x[3] - x[4]) ** 2);
  const bins = Array.from({ length: 10 }, () => ({ p: 0, freq: 0, n: 0 }));
  for (const x of r) {
    const b = bins[Math.min(9, Math.floor(x[1] * 10))];
    b.p += x[1];
    b.freq += x[4];
    b.n++;
  }
  const paired = r.filter((x) => Number.isFinite(x[2]));
  return {
    n,
    brier,
    brierClim,
    bss: n ? 1 - brier / brierClim : NaN,
    logLoss: avg((x) => logLoss(x[1], x[4])),
    logLossClim: avg((x) => logLoss(x[3], x[4])),
    hit: avg((x) => +((x[1] > 0.5) === (x[4] === 1))),
    cov68: avg((x) => x[6]),
    cov95: avg((x) => x[7]),
    vsClim: dieboldMariano(r.map((x) => logLoss(x[3], x[4]) - logLoss(x[1], x[4])), horizon - 1),
    vsChallenger: dieboldMariano(paired.map((x) => logLoss(x[1], x[4]) - logLoss(x[2], x[4])), horizon - 1),
    bins: bins.map((b) => ({ p: b.n ? b.p / b.n : NaN, freq: b.n ? b.freq / b.n : NaN, n: b.n })),
  };
}
