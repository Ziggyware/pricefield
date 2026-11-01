const $ = (s, r = document) => r.querySelector(s);
const el = (tag, cls, text) =>
  Object.assign(document.createElement(tag), cls ? { className: cls } : {}, text != null ? { textContent: text } : {});
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const ago = (t) => {
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 90 ? `${s.toFixed(0)} s` : s < 5400 ? `${(s / 60).toFixed(0)} min` : `${(s / 3600).toFixed(1)} h`;
};
const hm = (t) => new Date(t * 1000).toISOString().slice(11, 16);
const px = (v) => v >= 1000 ? v.toLocaleString("en-US", { maximumFractionDigits: 0 }) : v.toFixed(2);
const pct = (v) => `${(100 * v).toFixed(1)}%`;

function canvas(h) {
  const c = el("canvas"), d = devicePixelRatio || 1;
  c.style.height = `${h}px`;
  requestAnimationFrame(() => {
    c.width = c.clientWidth * d;
    c.height = h * d;
    c.draw?.(c.getContext("2d"), c.clientWidth, h, d);
  });
  return c;
}

/** Price path with the forecast distribution at each horizon: 95 % whisker, 68 % box, median tick. */
function fan(a, hs) {
  const c = canvas(220);
  c.draw = (g, W, H, d) => {
    g.scale(d, d);
    const cl = a.closes, t1 = a.t + 60, tEnd = t1 + Math.max(...hs) * 60, t0 = cl[0]?.[0] ?? t1 - 10800;
    const ys = cl.map((x) => x[1]);
    for (const h of hs) {
      const f = a.horizons[h].forecast;
      if (f) ys.push(f.price[0], f.price[4]);
    }
    const lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.08 || 1;
    const X = (t) => 8 + ((t - t0) / (tEnd - t0)) * (W - 70), Y = (v) => 8 + (1 - (v - lo + pad) / (hi - lo + 2 * pad)) * (H - 28);
    g.strokeStyle = css("--rail");
    g.fillStyle = css("--dim");
    g.font = `10px ${css("--mono")}`;
    g.lineWidth = 1;
    for (let k = 0; k <= 4; k++) {
      const v = lo - pad + ((hi - lo + 2 * pad) * k) / 4, y = Y(v);
      g.beginPath();
      g.moveTo(8, y);
      g.lineTo(W - 62, y);
      g.stroke();
      g.fillText(px(v), W - 58, y + 3);
    }
    for (let t = Math.ceil(t0 / 1800) * 1800; t <= tEnd; t += 1800) g.fillText(hm(t), X(t) - 14, H - 6);
    g.strokeStyle = css("--fg");
    g.lineWidth = 1.2;
    g.beginPath();
    cl.forEach(([t, v], k) => (k ? g.lineTo(X(t + 60), Y(v)) : g.moveTo(X(t + 60), Y(v))));
    g.stroke();
    g.setLineDash([2, 3]);
    g.strokeStyle = css("--dim");
    g.beginPath();
    g.moveTo(X(t1), 8);
    g.lineTo(X(t1), H - 20);
    g.stroke();
    g.setLineDash([]);
    for (const h of hs) {
      const f = a.horizons[h].forecast;
      if (!f) continue;
      const x = X(t1 + h * 60), up = f.p >= 0.5, col = up ? css("--up") : css("--down");
      g.strokeStyle = col;
      g.globalAlpha = 0.18;
      g.fillStyle = col;
      g.beginPath();
      g.moveTo(X(t1), Y(a.price));
      g.lineTo(x, Y(f.price[3]));
      g.lineTo(x, Y(f.price[1]));
      g.closePath();
      g.fill();
      g.globalAlpha = 1;
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x, Y(f.price[0]));
      g.lineTo(x, Y(f.price[4]));
      g.stroke();
      g.fillRect(x - 4, Y(f.price[3]), 8, Y(f.price[1]) - Y(f.price[3]));
      g.fillStyle = css("--void");
      g.fillRect(x - 5, Y(f.price[2]) - 1, 10, 2);
      g.fillStyle = css("--dim");
      g.fillText(`${h}m`, x - 8, Y(f.price[4]) + 12);
    }
  };
  return c;
}

/** Reliability: mean forecast vs observed frequency per probability decile; marker area ∝ count. */
function reliability(a, hs, colors) {
  const c = canvas(170);
  c.draw = (g, W, H, d) => {
    g.scale(d, d);
    const P = (v) => 24 + v * (W - 34), Q = (v) => H - 18 - v * (H - 28);
    g.strokeStyle = css("--rail");
    g.fillStyle = css("--dim");
    g.font = `10px ${css("--mono")}`;
    g.strokeRect(P(0), Q(1), P(1) - P(0), Q(0) - Q(1));
    g.beginPath();
    g.moveTo(P(0), Q(0));
    g.lineTo(P(1), Q(1));
    g.stroke();
    for (const v of [0, 0.5, 1]) {
      g.fillText(v.toFixed(1), P(v) - 8, H - 5);
      g.fillText(v.toFixed(1), 2, Q(v) + 3);
    }
    hs.forEach((h, k) => {
      const bins = a.horizons[h].metrics.bins, n = Math.max(1, ...bins.map((b) => b.n));
      g.fillStyle = colors[k];
      for (const b of bins) {
        if (b.n) {
          g.beginPath();
          g.arc(P(b.p), Q(b.freq), 1.5 + 5 * Math.sqrt(b.n / n), 0, 7);
          g.fill();
        }
      }
    });
  };
  return c;
}

function sparkline(curve, start, liveFrom) {
  const c = canvas(70);
  c.draw = (g, W, H, d) => {
    g.scale(d, d);
    if (curve.length < 2) return;
    const v = curve.map((x) => x[1]), lo = Math.min(start, ...v), hi = Math.max(start, ...v), t0 = curve[0][0], t1 = curve.at(-1)[0];
    const X = (t) => 4 + ((t - t0) / (t1 - t0 || 1)) * (W - 8), Y = (y) => 4 + (1 - (y - lo) / (hi - lo || 1)) * (H - 8);
    g.strokeStyle = css("--rail");
    g.beginPath();
    g.moveTo(4, Y(start));
    g.lineTo(W - 4, Y(start));
    g.stroke();
    if (liveFrom) {
      g.setLineDash([2, 3]);
      g.beginPath();
      g.moveTo(X(liveFrom), 4);
      g.lineTo(X(liveFrom), H - 4);
      g.stroke();
      g.setLineDash([]);
    }
    g.strokeStyle = css("--acid");
    g.lineWidth = 1.2;
    g.beginPath();
    curve.forEach(([t, y], k) => (k ? g.lineTo(X(t), Y(y)) : g.moveTo(X(t), Y(y))));
    g.stroke();
  };
  return c;
}

function horizonTable(a, hs) {
  const t = el("table"), head = el("tr");
  for (const h of ["h", "P(up)", "median", "68 %", "95 %", "E[r]", "BSS", "DM vs base", "n", "cov 68/95", "challenger"]) {
    head.append(el("th", null, h));
  }
  t.append(head);
  for (const h of hs) {
    const v = a.horizons[h], f = v.forecast, m = v.metrics, tr = el("tr");
    const bar = el("span", "pbar"), fill = el("i", f && f.p < 0.5 ? "dn" : "");
    if (f) {
      Object.assign(
        fill.style,
        f.p >= 0.5 ? { width: `${(f.p - 0.5) * 56}px` } : { width: `${(0.5 - f.p) * 56}px`, left: `${28 - (0.5 - f.p) * 56}px` },
      );
    }
    bar.append(fill);
    const p = el("td");
    p.append(bar, f ? pct(f.p) : "–");
    const dm = el(
      "td",
      m.vsClim.stat >= 1.96 ? "sig" : m.vsClim.stat <= -1.96 ? "neg" : "ns",
      m.n ? `${m.vsClim.stat.toFixed(2)} (p ${m.vsClim.p.toFixed(3)})` : "–",
    );
    const ch = v.challenger
      ? `η ${v.challenger.eta} λ ${v.challenger.l2}${v.challenger.off.length ? ` −${v.challenger.off.join(",")}` : ""} · DM ${
        m.vsChallenger.stat.toFixed(2)
      } n ${m.vsChallenger.n}`
      : "–";
    tr.append(
      el("td", null, `${h}m`),
      p,
      el("td", null, f ? px(f.price[2]) : "–"),
      el("td", null, f ? `${px(f.price[1])}–${px(f.price[3])}` : "–"),
      el("td", null, f ? `${px(f.price[0])}–${px(f.price[4])}` : "–"),
      el("td", null, f ? `${(f.er * 1e4).toFixed(1)} bp` : "–"),
      el("td", m.bss > 0 ? "" : "ns", m.n ? m.bss.toFixed(4) : "–"),
      dm,
      el("td", null, String(m.n)),
      el("td", null, m.n ? `${pct(m.cov68)} / ${pct(m.cov95)}` : "–"),
      el("td", "ns", ch),
    );
    t.append(tr);
  }
  return t;
}

/** Drivers of the 15-minute call: champion weight × current standardised feature, largest first. */
function drivers(a) {
  const w = a.horizons[15].champion.w,
    box = el("div", "drv"),
    rows = Object.entries(a.features ?? {}).map(([k, x]) => [k, w[k] * (k === "bias" ? 1 : x)]);
  rows.sort((p, q) => Math.abs(q[1]) - Math.abs(p[1]));
  const max = Math.max(1e-6, ...rows.map((r) => Math.abs(r[1])));
  for (const [k, v] of rows.slice(0, 10)) {
    const b = el("span", "hb"), i = el("i", v < 0 ? "dn" : "");
    Object.assign(i.style, v >= 0 ? { left: "50%", width: `${(50 * v) / max}%` } : { right: "50%", width: `${(-50 * v) / max}%` });
    b.append(i);
    box.append(el("span", null, k), b, el("span", "ro", v.toFixed(3)));
  }
  return box;
}

function paper(a, start) {
  const p = a.paper, s = p.stats, box = el("div");
  const line = el("div", "ro");
  line.innerHTML =
    `equity <b>${p.equity.toFixed(2)}</b> · ${s.n} trades · win ${s.n ? pct(s.wins / s.n) : "–"} · gross ${s.gross.toFixed(2)} · fees ${
      s.fees.toFixed(2)
    }` +
    (p.pos
      ? ` · <b>${p.pos.side > 0 ? "LONG" : "SHORT"}</b> ${p.pos.qty.toFixed(5)} @ ${px(p.pos.entry)} tp ${px(p.pos.tp)} sl ${px(p.pos.sl)}`
      : "") +
    (p.order ? ` · bid ${p.order.side > 0 ? "buy" : "sell"} @ ${px(p.order.limit)}` : "");
  const tr = el("div", "trades ro");
  for (const t of [...p.trades].reverse().slice(0, 8)) {
    tr.append(
      el(
        "div",
        null,
        `${hm(t.exitT)} ${t.side > 0 ? "L" : "S"} ${px(t.entry)}→${px(t.exit)} ${t.reason} ${t.pnl >= 0 ? "+" : ""}${t.pnl.toFixed(2)}${
          t.live ? "" : " (replay)"
        }`,
      ),
    );
  }
  box.append(line, sparkline(p.curve, start, p.liveFrom), tr);
  return box;
}

async function load() {
  let s;
  try {
    s = await fetch("/api/state").then((r) => r.json());
  } catch {
    $("#health").textContent = "server unreachable";
    return;
  }
  const hs = s.config.horizons, colors = [css("--acid"), css("--up"), css("--warn")], h = s.health;
  const errs = h ? Object.entries(h.assets).flatMap(([k, v]) => v.errors.map((e) => `${k} ${e}`)) : [];
  $("#health").innerHTML = h
    ? `tick <b>${ago(h.t)}</b> ago · ${h.ms} ms · router ${s.config.router ? "<b>keyed</b>" : '<span class="err">no key</span>'}` +
      (errs.length ? ` · <span class="err">${errs.join(" · ")}</span>` : "")
    : "waiting for first tick";
  const c = s.context;
  $("#ctx").innerHTML = c ? `<b>${c.regime}</b> · ${ago(c.t)} ago` : "no context yet";
  $("#assets").replaceChildren(...s.assets.map((a) => {
    const sec = el("section", "asset"), head = el("div", "head");
    head.append(
      el("b", null, a.asset),
      el("span", "px", a.price ? px(a.price) : "–"),
      el("span", "ro", a.sigma ? `σ1m ${(a.sigma * 1e4).toFixed(1)} bp · bar ${a.t ? hm(a.t) : "–"} UTC` : "warming up"),
    );
    const rel = el("div"), rh = el("h2");
    rh.innerHTML = `reliability ${hs.map((x, k) => `<span style="color:${colors[k]}">${x}m</span>`).join(" ")}`;
    rel.append(rh, reliability(a, hs, colors));
    const pp = el("div");
    pp.append(
      el("h2", null, `paper · ${s.config.paper.horizon}m · maker ${pct(s.config.paper.makerFee)} / taker ${pct(s.config.paper.takerFee)}`),
      paper(a, s.config.paper.startEquity),
    );
    const dv = el("div");
    dv.append(el("h2", null, "15m drivers (w·x)"), drivers(a));
    const g2 = el("div", "grid2");
    g2.append(rel, dv);
    const tw = el("div", "tw");
    tw.append(horizonTable(a, hs));
    sec.append(head, fan(a, hs), tw, g2, pp);
    return sec;
  }));
  const ctx = $("#context");
  ctx.innerHTML = c
    ? `<b>${c.regime}</b><br>${c.note}<br>${
      Object.entries(c.assets).map(([k, v]) =>
        `${k} sentiment <b>${v.sentiment.toFixed(2)}</b> event risk <b>${v.eventRisk.toFixed(2)}</b>`
      ).join("<br>")
    }<br>${c.model} · ${ago(c.t)} ago`
    : "The context agent runs every 15 minutes once ROUTER_API_KEY is set.";
  $("#journal").replaceChildren(...s.journal.map((j) => {
    const li = el("li");
    li.dataset.a = j.agent;
    li.append(el("span", "m", `${hm(j.t)} ${j.agent}${j.asset ? ` ${j.asset}` : ""}${j.model ? ` · ${j.model}` : ""} `), j.text);
    return li;
  }));
}

load();
setInterval(load, 15_000);
