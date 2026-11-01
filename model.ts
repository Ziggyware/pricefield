import { FEATURES, HYPER, HYPER_BOUNDS, NF, ZRES_N } from "./config.ts";
import { clamp, normInv, quantile, sigmoid } from "./stats.ts";

/**
 * Online L2-regularised logistic regression (Adagrad) on P(close[t+h] > close[t]), followed by a two-parameter
 * shrinkage layer p = σ(a + b·logit p_raw) fitted prequentially on raw logits recorded at forecast time.
 * b starts at 0, so a new model forecasts the base rate and moves away from it only as out-of-sample evidence
 * accrues; on a market with no exploitable structure b stays near 0 instead of broadcasting estimation noise.
 */
export interface Learner {
  w: number[];
  g2: number[];
  /** Shrinkage layer [a, b] and its information matrix [Haa, Hab, Hbb]. */
  cal: [number, number];
  ch: [number, number, number];
  eta: number;
  l2: number;
  /** Feature names held at zero (critic-proposed ablations). */
  off: string[];
  n: number;
}

export const newLearner = (h: Partial<Pick<Learner, "eta" | "l2" | "off">> = {}): Learner => ({
  w: new Array(NF).fill(0),
  g2: new Array(NF).fill(0),
  cal: [0, 0],
  ch: [1 / PRIOR_SD[0] ** 2, 0, 1 / PRIOR_SD[1] ** 2],
  eta: clamp(h.eta ?? HYPER.eta, ...HYPER_BOUNDS.eta),
  l2: clamp(h.l2 ?? HYPER.l2, ...HYPER_BOUNDS.l2),
  off: (h.off ?? []).filter((f) => FEATURES.includes(f as never) && f !== "bias"),
  n: 0,
});

const masked = (m: Learner, x: number[]) => x.map((v, j) => (m.off.includes(FEATURES[j]) ? 0 : v));

/** Raw logit of the first stage. */
export function logit(m: Learner, x: number[]): number {
  const xm = masked(m, x);
  return m.w.reduce((s, w, j) => s + w * xm[j], 0);
}

/** Calibrated probability from a raw logit. */
export const calibrated = (m: Learner, z: number) => sigmoid(m.cal[0] + m.cal[1] * z);

export const prob = (m: Learner, x: number[]) => calibrated(m, logit(m, x));

/** Gaussian prior on the shrinkage layer: a ~ N(0, 1), b ~ N(0, 0.5²) — trust in the first stage is earned. */
const PRIOR_SD = [1, 0.5] as const;
/** Information half-life of the shrinkage layer, in effective (overlap-weighted) observations. */
const CAL_HALFLIFE = 5000;

/**
 * Update on a matured outcome. `z` is the raw logit recorded when the forecast was made, so the shrinkage layer is
 * fitted out of sample. It takes an online Newton step on the MAP objective, each observation weighted by `rate` = 1/h
 * because overlapping h-minute labels sampled every minute carry about 1/h independent information each.
 */
export function learn(m: Learner, x: number[], z: number, y: 0 | 1, rate = 1): void {
  const pc = calibrated(m, z), keep = Math.pow(2, -rate / CAL_HALFLIFE), prec = pc * (1 - pc) * rate;
  const floor = [1 / PRIOR_SD[0] ** 2, 0, 1 / PRIOR_SD[1] ** 2];
  const H = m.ch.map((v, k) => floor[k] + keep * (v - floor[k])) as [number, number, number];
  H[0] += prec;
  H[1] += prec * z;
  H[2] += prec * z * z;
  const ga = rate * (pc - y), gb = rate * (pc - y) * z, det = H[0] * H[2] - H[1] * H[1];
  m.cal[0] -= (H[2] * ga - H[1] * gb) / det;
  m.cal[1] -= (H[0] * gb - H[1] * ga) / det;
  m.ch = H;
  const xm = masked(m, x), p = sigmoid(logit(m, x));
  for (let j = 0; j < NF; j++) {
    const g = (p - y) * xm[j] + (j === 0 ? 0 : m.l2 * m.w[j]);
    if (g === 0) continue;
    m.g2[j] += g * g;
    m.w[j] -= (m.eta * g) / (Math.sqrt(m.g2[j]) + 1e-8);
  }
  m.n++;
}

/** Log loss with probabilities clipped to [1e-6, 1 − 1e-6]. */
export const logLoss = (p: number, y: number) => -Math.log(y ? Math.max(p, 1e-6) : Math.max(1 - p, 1e-6));

export const QUANTS = [0.025, 0.16, 0.5, 0.84, 0.975] as const;

/**
 * Predictive distribution of the standardised return z = r_h / (σ√h) by filtered historical simulation:
 * the empirical distribution of past realised z, centred on its median, shifted by m so that P(z > 0) = p.
 * With fewer than 60 residuals a standard normal stands in. Returns the z quantiles in QUANTS order and E[z].
 */
export function predictive(zres: number[], p: number): { q: number[]; mean: number } {
  if (zres.length < 60) {
    const m = normInv(p);
    return { q: QUANTS.map((a) => m + normInv(a)), mean: m };
  }
  const med = quantile([...zres].sort((a, b) => a - b), 0.5);
  const c = zres.map((z) => z - med).sort((a, b) => a - b);
  const m = -quantile(c, 1 - p);
  return { q: QUANTS.map((a) => m + quantile(c, a)), mean: m + c.reduce((s, z) => s + z, 0) / c.length };
}

export function pushResidual(zres: number[], z: number): void {
  if (!Number.isFinite(z)) return;
  zres.push(z);
  if (zres.length > ZRES_N) zres.splice(0, zres.length - ZRES_N);
}

/** Beta(1,1) climatology: the no-skill baseline every forecast is scored against. */
export interface Clim {
  up: number;
  n: number;
}
export const climProb = (c: Clim) => (c.up + 1) / (c.n + 2);
