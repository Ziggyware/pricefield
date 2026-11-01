import { type Asset, VENUE } from "./config.ts";

export interface Bar {
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}
export interface Book {
  bids: [number, number][];
  asks: [number, number][];
}
export interface Trade {
  t: number;
  price: number;
  size: number;
  takerBuy: boolean;
}
export interface Snapshot {
  asset: Asset;
  cb: Bar[]; // Coinbase 1-minute bars, completed only, ascending
  gm: Bar[]; // Gemini
  cbBook: Book | null;
  gmBook: Book | null;
  trades: Trade[]; // both venues, ascending
  funding: number[]; // OKX funding history, oldest first, current last
  oi: [number, number][]; // OKX open interest (ms, USD), 5-minute, ascending
  errors: string[];
}

const UA = { "user-agent": "pricefield/1.0", accept: "application/json" };

async function get<T>(url: string, ms = 10_000): Promise<T> {
  const r = await fetch(url, { headers: UA, signal: AbortSignal.timeout(ms) });
  if (!r.ok) throw new Error(`${new URL(url).host} ${r.status}`);
  return await r.json() as T;
}

const completed = (bars: Bar[], nowSec: number) => bars.filter((b) => b.t + 60 <= nowSec).sort((a, b) => a.t - b.t);

/**
 * Coinbase public endpoints sit behind a CDN that caches by URL (candles max-age 300 s, books up to 4 h), so every
 * request carries a unique URL: explicit start/end for candles, a nonce elsewhere.
 */
const nonce = () => `_=${Date.now()}${Math.random().toString(36).slice(2, 8)}`;

/** Coinbase Exchange candles: [time, low, high, open, close, volume], newest first, ≤ 300 per request. */
export async function coinbaseBars(a: Asset, startSec: number, endSec: number): Promise<Bar[]> {
  const q = `&start=${new Date(startSec * 1e3).toISOString()}&end=${new Date(endSec * 1e3).toISOString()}`;
  const rows = await get<number[][]>(`https://api.exchange.coinbase.com/products/${VENUE.coinbase(a)}/candles?granularity=60${q}`);
  return rows.map(([t, l, h, o, c, v]) => ({ t, o, h, l, c, v }));
}

/** Gemini v2 candles: [ms, open, high, low, close, volume], newest first, ~24 h. */
async function geminiBars(a: Asset): Promise<Bar[]> {
  const rows = await get<number[][]>(`https://api.gemini.com/v2/candles/${VENUE.gemini(a)}/1m`);
  return rows.map(([t, o, h, l, c, v]) => ({ t: t / 1e3, o, h, l, c, v }));
}

const levels = (xs: { price: string; size?: string; amount?: string }[]) =>
  xs.map((x) => [+x.price, +(x.size ?? x.amount ?? 0)] as [number, number]);

async function coinbaseBook(a: Asset): Promise<Book> {
  const j = await get<{ pricebook: { bids: { price: string; size: string }[]; asks: { price: string; size: string }[] } }>(
    `https://api.coinbase.com/api/v3/brokerage/market/product_book?product_id=${VENUE.coinbase(a)}&limit=50&${nonce()}`,
  );
  return { bids: levels(j.pricebook.bids), asks: levels(j.pricebook.asks) };
}

async function geminiBook(a: Asset): Promise<Book> {
  const j = await get<{ bids: { price: string; amount: string }[]; asks: { price: string; amount: string }[] }>(
    `https://api.gemini.com/v1/book/${VENUE.gemini(a)}?limit_bids=50&limit_asks=50`,
  );
  return { bids: levels(j.bids), asks: levels(j.asks) };
}

/** Coinbase Exchange `side` is the maker's side, so the taker bought when side === "sell". */
async function coinbaseTrades(a: Asset): Promise<Trade[]> {
  const rows = await get<{ time: string; price: string; size: string; side: string }[]>(
    `https://api.exchange.coinbase.com/products/${VENUE.coinbase(a)}/trades?limit=1000&${nonce()}`,
  );
  return rows.map((r) => ({ t: Date.parse(r.time), price: +r.price, size: +r.size, takerBuy: r.side === "sell" }));
}

/** Gemini `type` is the taker's side. */
async function geminiTrades(a: Asset): Promise<Trade[]> {
  const rows = await get<{ timestampms: number; price: string; amount: string; type: string }[]>(
    `https://api.gemini.com/v1/trades/${VENUE.gemini(a)}?limit_trades=500`,
  );
  return rows.map((r) => ({ t: r.timestampms, price: +r.price, size: +r.amount, takerBuy: r.type === "buy" }));
}

async function funding(a: Asset): Promise<number[]> {
  const id = VENUE.okxSwap(a);
  const [hist, cur] = await Promise.all([
    get<{ data: { fundingRate: string }[] }>(`https://www.okx.com/api/v5/public/funding-rate-history?instId=${id}&limit=100`),
    get<{ data: { fundingRate: string }[] }>(`https://www.okx.com/api/v5/public/funding-rate?instId=${id}`),
  ]);
  return [...hist.data.map((d) => +d.fundingRate).reverse(), +cur.data[0].fundingRate];
}

async function openInterest(a: Asset): Promise<[number, number][]> {
  const j = await get<{ data: string[][] }>(`https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?ccy=${a}&period=5m`);
  return j.data.map(([t, oi]) => [+t, +oi] as [number, number]).reverse();
}

/** All sources in parallel; a failed source is recorded and replaced by an empty value, never fabricated. */
export async function snapshot(a: Asset, nowSec: number): Promise<Snapshot> {
  const errors: string[] = [];
  const safe = <T>(p: Promise<T>, empty: T, name: string) =>
    p.catch((e) => {
      errors.push(`${name}: ${e.message ?? e}`);
      return empty;
    });
  const [cb, gm, cbBook, gmBook, cbT, gmT, fund, oi] = await Promise.all([
    safe(coinbaseBars(a, nowSec - 299 * 60, nowSec), [] as Bar[], "coinbase candles"),
    safe(geminiBars(a), [] as Bar[], "gemini candles"),
    safe(coinbaseBook(a), null, "coinbase book"),
    safe(geminiBook(a), null, "gemini book"),
    safe(coinbaseTrades(a), [] as Trade[], "coinbase trades"),
    safe(geminiTrades(a), [] as Trade[], "gemini trades"),
    safe(funding(a), [] as number[], "okx funding"),
    safe(openInterest(a), [] as [number, number][], "okx open interest"),
  ]);
  return {
    asset: a,
    cb: completed(cb, nowSec),
    gm: completed(gm, nowSec),
    cbBook,
    gmBook,
    trades: [...cbT, ...gmT].sort((x, y) => x.t - y.t),
    funding: fund,
    oi,
    errors,
  };
}
