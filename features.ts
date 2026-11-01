import { CLIP, FEATURES, NF, SCALER_HALFLIFE } from "./config.ts";
import type { Bar, Book, Trade } from "./feeds.ts";
import { clamp, ewmaVol, hurst, kalmanSlope, mean, normCdf, std } from "./stats.ts";

/** Inputs that exist only live (books, trades, perp positioning, agent context). Absent ⇒ feature 0. */
export interface Live {
  gmClose?: Map<number, number>; // Gemini close by bar start (s)
  books?: (Book | null)[];
  trades?: Trade[];
  funding?: number[];
  oi?: [number, number][];
  sentiment?: number;
}

export interface Raw {
  x: number[]; // unscaled features, FEATURES order; NaN where the input is unavailable
  sigma: number; // EWMA 1-minute log-return σ
  price: number;
}

const MIN_BARS = 130;
const lr = (bars: Bar[]) => bars.slice(1).map((b, i) => Math.log(b.c / bars[i].c));

function imbalance(book: Book, mid: number, bp: number | null, levels: number): number {
  const side = (xs: [number, number][], below: boolean) =>
    xs.slice(0, bp === null ? levels : xs.length)
      .filter(([p]) => bp === null || (below ? p >= mid * (1 - bp) : p <= mid * (1 + bp)))
      .reduce((s, [, q]) => s + q, 0);
  const b = side(book.bids, true), a = side(book.asks, false);
  return b + a > 0 ? (b - a) / (b + a) : 0;
}

/** Features at bar `i` (inclusive) of ascending Coinbase bars. Returns null until enough history exists. */
export function rawFeatures(bars: Bar[], i: number, live: Live = {}): Raw | null {
  if (i + 1 < MIN_BARS) return null;
  const win = bars.slice(Math.max(0, i - 299), i + 1), r = lr(win), c = win.at(-1)!.c;
  const sigma = Math.max(ewmaVol(r, 60), 1e-6);
  const mom = (k: number) => Math.log(c / win[win.length - 1 - k].c) / (sigma * Math.sqrt(k));
  const r15 = Math.log(c / win[win.length - 16].c), r60 = Math.log(c / win[win.length - 61].c);
  const x = new Array(NF).fill(NaN);
  const set = (name: (typeof FEATURES)[number], v: number) => {
    if (Number.isFinite(v)) x[FEATURES.indexOf(name)] = v;
  };
  set("bias", 1);
  set("mom1", mom(1));
  set("mom5", mom(5));
  set("mom15", mom(15));
  set("mom60", mom(60));
  set("kalman_vel", kalmanSlope(win.slice(-240).map((b) => Math.log(b.c)), sigma) / sigma);
  set("vol_ratio", Math.log(Math.max(std(r.slice(-15)), 1e-9) / Math.max(std(r.slice(-120)), 1e-9)));
  set("hurst_mom", (hurst(r.slice(-64)) - 0.5) * Math.sign(r15));
  const g = live.gmClose?.get(win.at(-1)!.t);
  if (g) set("basis", Math.log(c / g) / sigma);

  // Bulk-volume classification VPIN over 50 bars (Easley, López de Prado, O'Hara 2012).
  const w50 = win.slice(-51), dp = w50.slice(1).map((b, k) => b.c - w50[k].c), sdp = std(dp) || 1;
  let tox = 0, vol = 0, signed = 0;
  dp.forEach((d, k) => {
    const v = w50[k + 1].v, buy = normCdf(d / sdp);
    tox += Math.abs(2 * buy - 1) * v;
    vol += v;
    if (k >= dp.length - 15) signed += (2 * buy - 1) * v;
  });
  if (vol > 0) set("vpin_flow", (tox / vol) * Math.sign(signed));

  const books = (live.books ?? []).filter((b): b is Book => !!b && b.bids.length > 0 && b.asks.length > 0);
  if (books.length) {
    set("obi", mean(books.map((b) => imbalance(b, (b.bids[0][0] + b.asks[0][0]) / 2, null, 5))));
    set("depth10", mean(books.map((b) => imbalance(b, (b.bids[0][0] + b.asks[0][0]) / 2, 0.001, 50))));
  }
  if (live.trades?.length) {
    const since = live.trades.at(-1)!.t - 5 * 60_000;
    let buy = 0, all = 0;
    for (const t of live.trades) {
      if (t.t < since) continue;
      const n = t.size * t.price;
      all += n;
      if (t.takerBuy) buy += n;
    }
    if (all > 0) set("flow", (2 * buy - all) / all);
  }
  if ((live.funding?.length ?? 0) >= 10) {
    const f = live.funding!, s = std(f.slice(0, -1));
    if (s > 0) set("funding_z", (f.at(-1)! - mean(f.slice(0, -1))) / s);
  }
  if ((live.oi?.length ?? 0) >= 48) {
    const oi = live.oi!.map(([, v]) => Math.log(v)), d = oi.slice(12).map((v, k) => v - oi[k]), s = std(d);
    if (s > 0) set("oi_mom", ((d.at(-1)! - mean(d)) / s) * Math.sign(r60));
  }
  if (live.sentiment) set("sentiment", live.sentiment);
  return { x, sigma, price: c };
}

/** Exponentially weighted standardisation per feature; each feature counts only the bars where it was observed. */
export interface Scaler {
  m: number[];
  v: number[];
  n: number[];
}
export const newScaler = (): Scaler => ({ m: new Array(NF).fill(0), v: new Array(NF).fill(1), n: new Array(NF).fill(0) });

/**
 * Standardise with the current state, then absorb the observation, so a vector never scales itself.
 * Unobserved (NaN) or not-yet-warm (< 30 observations) features map to 0: no evidence, no contribution.
 */
export function scale(s: Scaler, raw: number[]): number[] {
  const out = raw.map((
    v,
    j,
  ) => (j === 0 ? 1 : !Number.isFinite(v) || s.n[j] < 30 ? 0 : clamp((v - s.m[j]) / Math.sqrt(s.v[j] + 1e-12), -CLIP, CLIP)));
  const a = 1 - Math.pow(2, -1 / SCALER_HALFLIFE);
  for (let j = 1; j < NF; j++) {
    if (!Number.isFinite(raw[j])) continue;
    const a1 = s.n[j] < 1 / a ? 1 / (s.n[j] + 1) : a; // equal weights until the window fills
    const d = raw[j] - s.m[j];
    s.m[j] += a1 * d;
    s.v[j] = s.n[j] === 0 ? 0 : (1 - a1) * (s.v[j] + a1 * d * d);
    s.n[j]++;
  }
  return out;
}
