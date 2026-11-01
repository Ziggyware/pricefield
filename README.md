# PRICEFIELD

BTC and ETH probabilistic forecasts at 5, 15 and 60 minutes, scored live against a no-skill baseline, with a
persistent agent swarm on nokey-router and fee-accurate paper trading. Runs on Deno Deploy (KV + cron).

It reports how good it is. Every forecast is made before its outcome is known and scored afterwards; the dashboard
shows Brier skill against the base rate, a Diebold–Mariano test of whether that skill is real, and the empirical
coverage of the 68 % / 95 % bands. Short-horizon crypto returns are close to a random walk; expect skill near zero
unless a feature genuinely carries information, and treat anything without a significant DM statistic as noise.

## Pipeline (every minute)

1. **Feeds**: Coinbase + Gemini 1-minute candles, top-50 books and trades; OKX perp funding history and open interest.
   Coinbase's CDN caches public endpoints by URL, so every request carries a unique URL.
2. **Features** (16): vol-normalised momentum 1/5/15/60 m, local-linear-trend Kalman velocity, σ15/σ120, Anis–Lloyd
   Hurst × momentum sign, Coinbase–Gemini basis, top-5 book imbalance, ±10 bp depth imbalance, 5-minute taker flow,
   bulk-volume VPIN × flow sign, funding z, open-interest momentum, swarm sentiment. Features standardise online;
   a feature contributes only after 30 observations, so live-only inputs (books, trades, perps, sentiment) take
   30 minutes to start counting.
3. **Model** per asset × horizon: online L2 logistic regression (Adagrad), then a two-parameter shrinkage layer
   p = σ(a + b·logit) fitted out of sample by an online Newton step under a N(0, 0.5²) prior on b. A new model
   forecasts the base rate and moves away from it only as evidence accrues; overlapping labels are down-weighted 1/h.
4. **Distribution**: filtered historical simulation. Realised standardised returns z = r/(σ√h) are recentred and
   shifted so P(z > 0) = p, giving the median target and the 68 % / 95 % bands.
5. **Scoring**: matured forecasts update the model, the base-rate model, the residual memory and the scoreboard.
6. **Paper trading** on the 15-minute forecast: maker entry at the touch (filled only if the next bar trades through),
   taker exit at the 84 % / 2.5 % quantile or the horizon, fractional Kelly sizing. Gemini tier-0 fees (0.20 % maker,
   0.40 % taker) make a round trip 60 bp; it trades only when the expected move exceeds that.

The first tick per asset replays ~30 h of Coinbase history through the same code path, so the scoreboard starts full.

## Swarm (nokey-router)

| agent | cadence | does | authority |
|---|---|---|---|
| context | every 15 min | reads Cointelegraph + Decrypt headlines and market state → sentiment, event risk, regime | sentiment is a model feature whose weight is learned; event risk > 0.7 blocks new paper entries |
| analyst | every 5 min | explains flagged ≥ 4σ one-minute moves | journal only |
| critic | hourly | reads the scoreboard, proposes ≤ 2 challengers (η, λ, feature ablations) | none over champions: a challenger is promoted only when its paired live log loss beats the champion's by DM ≥ 1.96 over ≥ 300 forecasts, and retired otherwise |

All agent output is range-checked; failures go to the journal and never stop the minute pipeline.

## Deploy

```powershell
cd pricefield
deno deploy create --org ziggyware --app pricefield --source local --runtime-mode dynamic --entrypoint main.ts
deno deploy env add ROUTER_API_KEY "<nokey-router key>" --secret
deno deploy env add ADMIN_TOKEN "<any long random string>" --secret
deno deploy --prod
```
Crons register automatically (UTC; visible in the app's Cron tab). Optional: `ROUTER_URL` (defaults to
`https://nokey-router.ziggyware.deno.net`).

Local: `deno task dev` (KV in the default location, or set `KV_PATH`), `deno task test`.

## API

| route | |
|---|---|
| `GET /` | dashboard |
| `GET /api/state` | forecasts, scoreboards, model weights, paper book, context, journal, health |
| `POST /api/run/{tick,context,analyst,critic}` | run a step now; header `x-admin-token: $ADMIN_TOKEN` |

## Files

`config.ts` constants and bounds · `feeds.ts` exchange I/O · `features.ts` features + online scaler ·
`stats.ts` numerics (Φ, Φ⁻¹, ln Γ, Kalman, Hurst) · `model.ts` learner, shrinkage, FHS · `score.ts` records, DM test,
reliability · `engine.ts` per-bar step · `paper.ts` simulated execution · `swarm.ts` router client + agents ·
`store.ts` KV layout, journal, locks · `tick.ts` minute cron · `main.ts` crons + HTTP · `ui/` dashboard · `test.ts`.
