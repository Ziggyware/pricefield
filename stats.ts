/** Numerical primitives. Deterministic; no hidden state. */

/** Φ(x) via erf, Abramowitz–Stegun 7.1.26 (|ε| < 1.5e-7). */
export function normCdf(x: number): number {
  const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.3275911 * z);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

/** Φ⁻¹(p), Acklam's rational approximation with one Halley refinement (|ε| ≈ 1e-15). */
export function normInv(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const q = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
  let x: number;
  if (q < 0.02425) {
    const r = Math.sqrt(-2 * Math.log(q));
    x = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1);
  } else if (q > 1 - 0.02425) {
    const r = Math.sqrt(-2 * Math.log(1 - q));
    x = -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1);
  } else {
    const r = q - 0.5, s = r * r;
    x = (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r /
      (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1);
  }
  const e = normCdf(x) - q, u = e * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2);
  return x - u / (1 + (x * u) / 2);
}

/** ln Γ(x), Lanczos g = 7, n = 9. */
export function lgamma(x: number): number {
  const g = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let s = g[0];
  for (let i = 1; i < 9; i++) s += g[i] / (x + i);
  const t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(s);
}

export const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / Math.max(xs.length, 1);
export function std(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1));
}
export const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-clamp(z, -30, 30)));

/** EWMA volatility of a return series with the given half-life (observations). */
export function ewmaVol(r: number[], halfLife: number): number {
  const lam = Math.pow(2, -1 / halfLife);
  let v = r.slice(0, 20).reduce((s, x) => s + x * x, 0) / Math.max(Math.min(20, r.length), 1);
  for (const x of r) v = lam * v + (1 - lam) * x * x;
  return Math.sqrt(v);
}

/** Linear-interpolated empirical quantile of an ascending array. */
export function quantile(sorted: number[], p: number): number {
  if (!sorted.length) return NaN;
  const h = (sorted.length - 1) * clamp(p, 0, 1), lo = Math.floor(h);
  return sorted[lo] + (h - lo) * ((sorted[Math.min(lo + 1, sorted.length - 1)]) - sorted[lo]);
}

/**
 * Local-linear-trend Kalman filter on log price: state [level, slope], F = [[1,1],[0,1]], H = [1,0].
 * R = σ² (1-minute return variance); Q = diag(0, (κσ)²). Returns the filtered slope per minute.
 */
export function kalmanSlope(logp: number[], sigma: number, kappa = 0.05): number {
  const R = sigma * sigma, q = (kappa * sigma) ** 2;
  let x0 = logp[0], x1 = 0, p00 = R, p01 = 0, p11 = R;
  for (let i = 1; i < logp.length; i++) {
    // predict
    x0 += x1;
    const a00 = p00 + 2 * p01 + p11, a01 = p01 + p11, a11 = p11 + q;
    // update
    const s = a00 + R, k0 = a00 / s, k1 = a01 / s, e = logp[i] - x0;
    x0 += k0 * e;
    x1 += k1 * e;
    p00 = (1 - k0) * a00;
    p01 = (1 - k0) * a01;
    p11 = a11 - k1 * a01;
  }
  return x1;
}

/** Anis–Lloyd expected R/S for a white-noise sample of size n. */
function expectedRS(n: number): number {
  let s = 0;
  for (let i = 1; i < n; i++) s += Math.sqrt((n - i) / i);
  const lead = n <= 340 ? Math.exp(lgamma((n - 1) / 2) - lgamma(n / 2)) / Math.sqrt(Math.PI) : 1 / Math.sqrt((n * Math.PI) / 2);
  return ((n - 0.5) / n) * lead * s;
}

/** Hurst exponent by rescaled range with Anis–Lloyd small-sample correction: 0.5 for white noise. */
export function hurst(r: number[]): number {
  const sizes = [8, 16, 32, 64].filter((n) => n <= r.length);
  if (sizes.length < 2) return 0.5;
  const xs: number[] = [], ys: number[] = [];
  for (const n of sizes) {
    const rs: number[] = [];
    for (let s = r.length - n; s >= 0 && rs.length < Math.floor(r.length / n); s -= n) {
      const w = r.slice(s, s + n), m = mean(w), sd = std(w);
      if (sd === 0) continue;
      let cum = 0, hi = -Infinity, lo = Infinity;
      for (const x of w) {
        cum += x - m;
        hi = Math.max(hi, cum);
        lo = Math.min(lo, cum);
      }
      rs.push((hi - lo) / sd);
    }
    if (!rs.length) continue;
    xs.push(Math.log(n));
    ys.push(Math.log(mean(rs)) - Math.log(expectedRS(n)));
  }
  if (xs.length < 2) return 0.5;
  const mx = mean(xs), my = mean(ys);
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  return 0.5 + num / den;
}
