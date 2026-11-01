import { type Asset, type Horizon, HORIZONS } from "./config.ts";
import { type AssetState, newAssetState } from "./engine.ts";

/** Deno KV; on Deno Deploy the platform database, locally KV_PATH (or the default). */
export const kv = await Deno.openKv(Deno.env.get("KV_PATH") || undefined);

const f32 = (a: Float32Array) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
const fromBytes = (u: Uint8Array | null) => (u ? new Float32Array(u.slice().buffer) : new Float32Array(0));
const round = (x: number) => Math.round(x * 1e6) / 1e6;

/**
 * One asset is stored as nine keys written atomically (each ≤ 64 KiB):
 * core JSON, pending JSON, paper JSON, and per horizon the score buffer and FHS residuals as f32.
 */
export async function loadAsset(a: Asset): Promise<AssetState | null> {
  const keys = [
    ["a", a, "core"],
    ["a", a, "pending"],
    ["a", a, "paper"],
    ...HORIZONS.flatMap((h) => [["a", a, "score", h], ["a", a, "zres", h]]),
  ];
  const got = await kv.getMany(keys as Deno.KvKey[]);
  if (!got[0].value) return null;
  const S = newAssetState(a);
  Object.assign(S, got[0].value as object);
  S.pending = (got[1].value as AssetState["pending"]) ?? [];
  S.paper = (got[2].value as AssetState["paper"]) ?? S.paper;
  HORIZONS.forEach((h, k) => {
    S.scores[h] = fromBytes(got[3 + 2 * k].value as Uint8Array | null);
    S.zres[h] = Array.from(fromBytes(got[4 + 2 * k].value as Uint8Array | null));
  });
  S.events = [];
  return S;
}

export async function saveAsset(S: AssetState): Promise<void> {
  const a = S.asset;
  const { asset, lastBar, scaler, models, latest } = S;
  const tx = kv.atomic()
    .set(["a", a, "core"], { asset, lastBar, scaler, models, latest })
    .set(["a", a, "pending"], S.pending)
    .set(["a", a, "paper"], { ...S.paper, curve: S.paper.curve.map(([t, e]) => [t, round(e)]) });
  for (const h of HORIZONS as readonly Horizon[]) {
    tx.set(["a", a, "score", h], f32(S.scores[h])).set(["a", a, "zres", h], f32(new Float32Array(S.zres[h])));
  }
  const r = await tx.commit();
  if (!r.ok) throw new Error(`KV commit failed for ${a}`);
}

export interface JournalEntry {
  t: number;
  agent: string;
  text: string;
  model?: string;
  asset?: string;
}

/** Append-only journal capped at 200 entries. */
export async function journal(e: JournalEntry): Promise<void> {
  for (let attempt = 0; attempt < 5; attempt++) {
    const cur = await kv.get<JournalEntry[]>(["journal"]);
    const next = [...(cur.value ?? []), e].slice(-200);
    if ((await kv.atomic().check(cur).set(["journal"], next).commit()).ok) return;
  }
}

export async function getJournal(): Promise<JournalEntry[]> {
  return (await kv.get<JournalEntry[]>(["journal"])).value ?? [];
}

/**
 * Per-asset mutual exclusion across concurrently running crons: a KV key created only if absent, expiring after
 * `ttlMs` so a crashed holder cannot wedge the asset. Throws if the lock cannot be taken within `waitMs`.
 */
export async function withLock<T>(name: string, fn: () => Promise<T>, ttlMs = 120_000, waitMs = 45_000): Promise<T> {
  const key = ["lock", name], id = crypto.randomUUID(), until = Date.now() + waitMs;
  while (!(await kv.atomic().check({ key, versionstamp: null }).set(key, id, { expireIn: ttlMs }).commit()).ok) {
    if (Date.now() > until) throw new Error(`lock ${name} busy`);
    await new Promise((r) => setTimeout(r, 500));
  }
  try {
    return await fn();
  } finally {
    const cur = await kv.get(key);
    if (cur.value === id) await kv.atomic().check(cur).delete(key).commit();
  }
}
