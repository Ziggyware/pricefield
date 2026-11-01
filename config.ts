/** PRICEFIELD configuration. Every tunable lives here; the critic agent may only change the bounded subset in HYPER_BOUNDS. */

export const ASSETS = ["BTC", "ETH"] as const;
export type Asset = (typeof ASSETS)[number];

/** Forecast horizons in minutes. */
export const HORIZONS = [5, 15, 60] as const;
export type Horizon = (typeof HORIZONS)[number];

export const VENUE = {
  coinbase: (a: Asset) => `${a}-USD`,
  gemini: (a: Asset) => `${a.toLowerCase()}usd`,
  okxSwap: (a: Asset) => `${a}-USDT-SWAP`,
};

/** Feature vector layout. Index 0 is the intercept. */
export const FEATURES = [
  "bias",
  "mom1",
  "mom5",
  "mom15",
  "mom60", // vol-normalised log returns over k minutes
  "kalman_vel", // local-linear-trend Kalman velocity / σ
  "vol_ratio", // ln(σ15 / σ120)
  "hurst_mom", // (H_AL − 0.5) × sign(r15): persistence-weighted momentum
  "basis", // Coinbase − Gemini mid, in 1-minute σ units (cross-venue lead-lag)
  "obi", // top-5 order-book imbalance, venue mean
  "depth10", // ±10 bp depth imbalance, venue mean
  "flow", // taker volume imbalance, last 5 minutes, both venues
  "vpin_flow", // bulk-volume VPIN × sign(BVC imbalance)
  "funding_z", // OKX perp funding rate z-score over its history
  "oi_mom", // 1h open-interest change z × sign(r60)
  "sentiment", // swarm context agent, decayed (half-life 2 h)
] as const;
export const NF = FEATURES.length;

/** Online logistic regression defaults (Adagrad, L2). */
export const HYPER = { eta: 0.05, l2: 1e-4 };
export const HYPER_BOUNDS = { eta: [0.005, 0.5], l2: [0, 0.01] } as const;

/** Exponential forgetting for feature standardisation and climatology (≈ 1 week of minutes). */
export const SCALER_HALFLIFE = 7 * 24 * 60;
/** Feature clip after standardisation. */
export const CLIP = 4;

/** Filtered-historical-simulation residual memory per asset × horizon. */
export const ZRES_N = 720;
/** Scored forecast memory per asset × horizon (reliability, DM test). */
export const SCORE_N = 1000;

/** Challenger promotion: minimum paired forecasts and two-sided DM significance. */
export const PROMOTE_MIN_N = 300;
export const PROMOTE_Z = 1.96;
export const CHALLENGER_MAX_N = 3000;

/** Paper trading on the 15-minute forecast. Gemini ActiveTrader tier-0 fees: maker entry, taker exit (never conflated). */
export const PAPER = {
  horizon: 15 as Horizon,
  startEquity: 10_000,
  makerFee: 0.002,
  takerFee: 0.004,
  /** Fraction of full Kelly. */
  kellyScale: 0.25,
  maxLeverage: 1,
  /** No new entries while the context agent's event risk exceeds this. */
  maxEventRisk: 0.7,
};

export const ROUTER_URL = Deno.env.get("ROUTER_URL") ?? "https://nokey-router.ziggyware.deno.net";
export const ROUTER_KEY = Deno.env.get("ROUTER_API_KEY") ?? "";

/** News feeds read by the context agent. */
export const NEWS_FEEDS = ["https://cointelegraph.com/rss", "https://decrypt.co/feed"];
