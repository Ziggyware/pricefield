import { type Asset, CHALLENGER_MAX_N, FEATURES, type Horizon, HORIZONS, PAPER, PROMOTE_MIN_N, PROMOTE_Z } from "./config.ts";
import type { Bar } from "./feeds.ts";
import { type Live, newScaler, rawFeatures, scale, type Scaler } from "./features.ts";
import {
  calibrated,
  type Clim,
  climProb,
  learn,
  type Learner,
  logit,
  logLoss,
  newLearner,
  predictive,
  pushResidual,
  QUANTS,
} from "./model.ts";
import { newPaper, type Paper, stepPaper } from "./paper.ts";
import { dieboldMariano, metrics, pushRecord, REC, rows } from "./score.ts";

export interface Pending {
  t: number;
  h: Horizon;
  x: number[];
  z: number;
  zc: number;
  p: number;
  pc: number;
  pclim: number;
  sigma: number;
  price: number;
  q: number[];
}
export interface Model {
  champ: Learner;
  chal: Learner | null;
  chalSince: number;
  chalWhy: string;
  clim: Clim;
}
export interface Forecast {
  p: number;
  pc: number | null;
  pclim: number;
  er: number;
  price: number[];
  qz: number[];
}

/** Everything learned for one asset. Score buffers and residuals live beside it (binary). */
export interface AssetState {
  asset: Asset;
  lastBar: number;
  scaler: Scaler;
  models: Record<Horizon, Model>;
  pending: Pending[];
  zres: Record<Horizon, number[]>;
  scores: Record<Horizon, Float32Array>;
  paper: Paper;
  latest:
    | { t: number; price: number; sigma: number; x: number[]; f: Partial<Record<Horizon, Forecast>>; closes: [number, number][] }
    | null;
  events: string[]; // promotions / retirements this step, for the journal
}

export function newAssetState(asset: Asset): AssetState {
  const rec = <T>(f: () => T) => Object.fromEntries(HORIZONS.map((h) => [h, f()])) as Record<Horizon, T>;
  return {
    asset,
    lastBar: 0,
    scaler: newScaler(),
    models: rec(() => ({ champ: newLearner(), chal: null, chalSince: 0, chalWhy: "", clim: { up: 0, n: 0 } })),
    pending: [],
    zres: rec(() => []),
    scores: rec(() => new Float32Array(0)),
    paper: newPaper(),
    latest: null,
    events: [],
  };
}

/** Promote a challenger whose paired log loss is significantly lower; retire one that never gets there. */
function judge(S: AssetState, h: Horizon) {
  const m = S.models[h];
  if (!m.chal) return;
  const paired = rows(S.scores[h]).filter((r) => r[0] >= m.chalSince && Number.isFinite(r[2]));
  const dm = dieboldMariano(paired.map((r) => logLoss(r[1], r[4]) - logLoss(r[2], r[4])), h - 1);
  if (dm.n >= PROMOTE_MIN_N && dm.stat >= PROMOTE_Z) {
    S.events.push(`${S.asset} ${h}m: challenger promoted (DM ${dm.stat.toFixed(2)}, p ${dm.p.toFixed(4)}, n ${dm.n}) — ${m.chalWhy}`);
    m.champ = m.chal;
    m.chal = null;
  } else if (dm.n >= CHALLENGER_MAX_N || (dm.n >= PROMOTE_MIN_N && dm.stat <= -PROMOTE_Z)) {
    S.events.push(`${S.asset} ${h}m: challenger retired (DM ${dm.stat.toFixed(2)}, n ${dm.n}) — ${m.chalWhy}`);
    m.chal = null;
  }
}

/** Advance one completed bar: score and learn from matured forecasts, then forecast from this bar. */
export function step(S: AssetState, bars: Bar[], i: number, live: Live, eventRisk: number, isLive: boolean): void {
  const bar = bars[i];
  const closeAt = new Map(bars.map((b) => [b.t, b.c]));
  const keep: Pending[] = [];
  for (const f of S.pending) {
    const target = f.t + f.h * 60;
    if (target > bar.t) {
      keep.push(f);
      continue;
    }
    const c = closeAt.get(target);
    if (c === undefined) continue; // gap in the tape: drop rather than guess
    const y = c > f.price ? 1 : 0, z = Math.log(c / f.price) / (f.sigma * Math.sqrt(f.h));
    S.scores[f.h] = pushRecord(S.scores[f.h], [
      f.t,
      f.p,
      f.pc,
      f.pclim,
      y,
      z,
      +(z >= f.q[1] && z <= f.q[3]),
      +(z >= f.q[0] && z <= f.q[4]),
    ]);
    const m = S.models[f.h];
    learn(m.champ, f.x, f.z, y, 1 / f.h);
    if (m.chal) learn(m.chal, f.x, Number.isFinite(f.zc) ? f.zc : logit(m.chal, f.x), y, 1 / f.h);
    m.clim.up += y;
    m.clim.n++;
    pushResidual(S.zres[f.h], z);
    judge(S, f.h);
  }
  S.pending = keep;
  S.lastBar = bar.t;

  const raw = rawFeatures(bars, i, live);
  if (!raw) {
    stepPaper(S.paper, bar, null, isLive);
    return;
  }
  const x = scale(S.scaler, raw.x);
  const fc: Partial<Record<Horizon, Forecast>> = {};
  for (const h of HORIZONS) {
    const m = S.models[h], z = logit(m.champ, x), zc = m.chal ? logit(m.chal, x) : NaN;
    const p = calibrated(m.champ, z), pc = m.chal ? calibrated(m.chal, zc) : NaN, pclim = climProb(m.clim);
    const d = predictive(S.zres[h], p), s = raw.sigma * Math.sqrt(h);
    S.pending.push({
      t: bar.t,
      h,
      x: x.map((v) => Math.round(v * 1e4) / 1e4),
      z,
      zc,
      p,
      pc,
      pclim,
      sigma: raw.sigma,
      price: raw.price,
      q: d.q,
    });
    fc[h] = { p, pc: m.chal ? pc : null, pclim, er: d.mean * s, price: d.q.map((q) => raw.price * Math.exp(q * s)), qz: d.q };
  }
  S.latest = {
    t: bar.t,
    price: raw.price,
    sigma: raw.sigma,
    x,
    f: fc,
    closes: bars.slice(Math.max(0, i - 179), i + 1).map((b) => [b.t, b.c]),
  };
  const tf = fc[PAPER.horizon]!, sTrade = raw.sigma * Math.sqrt(PAPER.horizon);
  const book = live.books?.find((b) => b && b.bids.length && b.asks.length);
  stepPaper(S.paper, bar, {
    p: tf.p,
    er: tf.er,
    sd: Math.max(sTrade * (tf.qz[3] - tf.qz[1]) / 2, 1e-9),
    tp: tf.price[3],
    sl: tf.price[0],
    tpShort: tf.price[1],
    slShort: tf.price[4],
    touch: book ? { bid: book.bids[0][0], ask: book.asks[0][0] } : { bid: bar.c, ask: bar.c },
    eventRisk,
  }, isLive);
}

/** Dashboard / agent view of an asset. */
export function summary(S: AssetState) {
  const per = Object.fromEntries(HORIZONS.map((h) => {
    const m = S.models[h];
    return [h, {
      metrics: metrics(S.scores[h], h),
      champion: {
        eta: m.champ.eta,
        l2: m.champ.l2,
        off: m.champ.off,
        n: m.champ.n,
        shrink: m.champ.cal.map((v) => +v.toFixed(4)),
        w: Object.fromEntries(FEATURES.map((f, j) => [f, +m.champ.w[j].toFixed(4)])),
      },
      challenger: m.chal ? { eta: m.chal.eta, l2: m.chal.l2, off: m.chal.off, since: m.chalSince, why: m.chalWhy } : null,
      forecast: S.latest?.f[h] ?? null,
      quantiles: QUANTS,
    }];
  }));
  const pp = S.paper;
  return {
    asset: S.asset,
    t: S.latest?.t ?? null,
    price: S.latest?.price ?? null,
    sigma: S.latest?.sigma ?? null,
    closes: S.latest?.closes ?? [],
    features: S.latest ? Object.fromEntries(FEATURES.map((f, j) => [f, +S.latest!.x[j].toFixed(3)])) : null,
    horizons: per,
    paper: {
      equity: pp.equity,
      pos: pp.pos,
      order: pp.order,
      trades: pp.trades.slice(-20),
      curve: pp.curve,
      liveFrom: pp.liveFrom,
      stats: pp.stats,
    },
    recordWidth: REC,
  };
}
