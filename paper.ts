import { PAPER } from "./config.ts";
import type { Bar } from "./feeds.ts";
import { clamp } from "./stats.ts";

/**
 * Paper trading on the 15-minute forecast. Entry: maker limit at the touch, filled only if the next bar trades
 * through it. Exit: taker at TP / SL (SL assumed first when one bar spans both) or at the horizon close.
 * Size: fractional Kelly for log utility, f = scale × (E[r] − costs) / Var[r], capped at maxLeverage.
 */
export interface Position {
  side: 1 | -1;
  qty: number;
  entry: number;
  tp: number;
  sl: number;
  t: number;
  entryFee: number;
}
export interface Order {
  side: 1 | -1;
  qty: number;
  limit: number;
  tp: number;
  sl: number;
  t: number;
}
export interface Trade {
  side: 1 | -1;
  entryT: number;
  exitT: number;
  entry: number;
  exit: number;
  pnl: number;
  fees: number;
  reason: string;
  live: boolean;
}
export interface Paper {
  equity: number;
  pos: Position | null;
  order: Order | null;
  trades: Trade[];
  curve: [number, number][];
  liveFrom: number | null;
  stats: { n: number; wins: number; fees: number; gross: number };
}

export const newPaper = (): Paper => ({
  equity: PAPER.startEquity,
  pos: null,
  order: null,
  trades: [],
  curve: [],
  liveFrom: null,
  stats: { n: 0, wins: 0, fees: 0, gross: 0 },
});

export interface Signal {
  p: number; // P(up) at the trading horizon
  er: number; // expected log return
  sd: number; // predictive std of the log return
  tp: number; // price at the predictive 84 % (long) / 16 % (short) quantile
  sl: number; // price at the 2.5 % (long) / 97.5 % (short) quantile
  tpShort: number;
  slShort: number;
  touch: { bid: number; ask: number };
  eventRisk: number;
}

function close(pp: Paper, bar: Bar, price: number, reason: string, live: boolean) {
  const p = pp.pos!;
  const fee = Math.abs(p.qty) * price * PAPER.takerFee;
  const gross = p.side * p.qty * (price - p.entry);
  pp.equity += gross - fee;
  pp.stats.n++;
  pp.stats.fees += fee + p.entryFee;
  pp.stats.gross += gross;
  if (gross - fee - p.entryFee > 0) pp.stats.wins++;
  pp.trades.push({
    side: p.side,
    entryT: p.t,
    exitT: bar.t,
    entry: p.entry,
    exit: price,
    pnl: gross - fee - p.entryFee,
    fees: fee + p.entryFee,
    reason,
    live,
  });
  if (pp.trades.length > 50) pp.trades.shift();
  pp.pos = null;
}

/** Advance one completed bar; `sig` is the forecast made at this bar's close (null while warming). */
export function stepPaper(pp: Paper, bar: Bar, sig: Signal | null, live: boolean): void {
  if (pp.order) {
    const o = pp.order;
    const filled = o.side === 1 ? bar.l < o.limit : bar.h > o.limit;
    if (filled) {
      const entryFee = o.qty * o.limit * PAPER.makerFee;
      pp.equity -= entryFee;
      pp.pos = { side: o.side, qty: o.qty, entry: o.limit, tp: o.tp, sl: o.sl, t: bar.t, entryFee };
    }
    pp.order = null;
  }
  if (pp.pos) {
    const p = pp.pos;
    const hitSL = p.side === 1 ? bar.l <= p.sl : bar.h >= p.sl;
    const hitTP = p.side === 1 ? bar.h >= p.tp : bar.l <= p.tp;
    if (hitSL) close(pp, bar, p.sl, "stop", live);
    else if (hitTP) close(pp, bar, p.tp, "target", live);
    else if (bar.t >= p.t + (PAPER.horizon - 1) * 60) close(pp, bar, bar.c, "horizon", live);
  }
  if (!pp.pos && sig && sig.eventRisk <= PAPER.maxEventRisk) {
    const cost = PAPER.makerFee + PAPER.takerFee;
    const side: 0 | 1 | -1 = sig.er - cost > 0 && sig.p > 0.5 ? 1 : -sig.er - cost > 0 && sig.p < 0.5 ? -1 : 0;
    if (side) {
      const f = clamp((PAPER.kellyScale * (Math.abs(sig.er) - cost)) / (sig.sd * sig.sd), 0, PAPER.maxLeverage);
      const limit = side === 1 ? sig.touch.bid : sig.touch.ask;
      if (f > 0 && limit > 0) {
        pp.order = {
          side,
          qty: (f * pp.equity) / limit,
          limit,
          tp: side === 1 ? sig.tp : sig.tpShort,
          sl: side === 1 ? sig.sl : sig.slShort,
          t: bar.t,
        };
      }
    }
  }
  if (live && pp.liveFrom === null) pp.liveFrom = bar.t;
  const mark = pp.pos ? pp.equity + pp.pos.side * pp.pos.qty * (bar.c - pp.pos.entry) : pp.equity;
  if (!pp.curve.length || bar.t - pp.curve.at(-1)![0] >= 300) pp.curve.push([bar.t, mark]);
  if (pp.curve.length > 1000) pp.curve.splice(0, pp.curve.length - 1000);
}
