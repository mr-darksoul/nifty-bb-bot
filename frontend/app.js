// ── Backend URL ───────────────────────────────────────────────────────────────
// Cloud Run (asia-south1) backend serving the bot API + WebSocket.
const BACKEND_URL = "https://nifty-bb-bot-950128522459.asia-south1.run.app";
const API_TOKEN_STORAGE_KEY = "nifty_bb_api_token";
let apiTokenPrompted = false;

// ── State ─────────────────────────────────────────────────────────────────────
let ws = null;
let wsRetryTimer = null;
let prevPrice = 0;
let candleSeries = null;
let bbUpperSeries = null;
let bbMidSeries   = null;
let bbLowSeries   = null;
let btEquitySeries = null;
let tvChart = null;
let btEquityChart = null;

// ── Utility ───────────────────────────────────────────────────────────────────

function fmt(n, d = 2)   { return n == null ? "—" : Number(n).toFixed(d); }
function fmtPct(n)        { return n == null ? "—" : (Number(n) * 100).toFixed(1) + "%"; }
function fmtInr(n)        { return n == null ? "—" : "₹" + Number(n).toLocaleString("en-IN", {minimumFractionDigits:2, maximumFractionDigits:2}); }
function el(id)           { return document.getElementById(id); }

function getApiToken() {
  let token = localStorage.getItem(API_TOKEN_STORAGE_KEY) || "";
  if (!token && !apiTokenPrompted) {
    apiTokenPrompted = true;
    token = window.prompt("Enter dashboard API token") || "";
    token = token.trim();
    if (token) localStorage.setItem(API_TOKEN_STORAGE_KEY, token);
  }
  return token;
}

function authHeaders() {
  const token = getApiToken();
  return token ? { "X-API-Token": token } : {};
}

function wsUrl() {
  const token = encodeURIComponent(getApiToken());
  return BACKEND_URL.replace(/^http/, "ws") + "/ws/live?token=" + token;
}

function toast(msg, type = "info") {
  const c = el("toast-container");
  const t = document.createElement("div");
  t.className = "toast " + (type === "error" ? "error" : type === "ok" ? "success" : "");
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

function tsToUnix(iso) {
  return Math.floor(new Date(iso).getTime() / 1000);
}

// ── Bollinger %b colour ───────────────────────────────────────────────────────

function pbColour(pb) {
  const v = parseFloat(pb);
  if (isNaN(v)) return "var(--muted)";
  if (v < 0.1)  return "#f85149";   // oversold → red
  if (v > 0.9)  return "#f85149";   // overbought → red
  if (v < 0.3)  return "#d29922";   // approaching oversold → yellow
  if (v > 0.7)  return "#d29922";   // approaching overbought → yellow
  return "#3fb950";                  // neutral → green
}

function scoreColour(s) {
  const v = parseFloat(s);
  if (isNaN(v)) return "var(--muted)";
  if (v >= 0.75) return "#3fb950";
  if (v >= 0.60) return "#d29922";
  return "#f85149";
}

// ── Regime badge ──────────────────────────────────────────────────────────────

function regimeBadgeClass(name) {
  if (!name) return "muted";
  const n = name.toUpperCase();
  if (n === "CHOPPY")         return "green";
  if (n === "TRENDING_UP")    return "yellow";
  if (n === "TRENDING_DOWN")  return "yellow";
  return "muted";
}

function applyBadge(id, text, cls) {
  const b = el(id);
  if (!b) return;
  b.textContent = text;
  b.className = "badge " + cls;
}

// ── Header updates ────────────────────────────────────────────────────────────

function updateHeader(data) {
  const price = data.price || data.nifty_price || 0;
  el("hdr-price").textContent = fmt(price, 2);

  if (prevPrice && price) {
    const chg = price - prevPrice;
    const pct  = (chg / prevPrice * 100).toFixed(2);
    const sign = chg >= 0 ? "+" : "";
    const chgEl = el("hdr-change");
    chgEl.textContent = `${sign}${fmt(chg,2)} (${sign}${pct}%)`;
    chgEl.className   = "price-change " + (chg >= 0 ? "up" : "down");
  }
  if (price) prevPrice = price;
}

function updateBotBadge(running, marketOpen) {
  applyBadge("badge-market", marketOpen ? "OPEN" : "CLOSED", marketOpen ? "green" : "muted");
  applyBadge("badge-bot",    running    ? "BOT ON" : "BOT OFF", running ? "green" : "muted");
}

// ── Signal panel ──────────────────────────────────────────────────────────────

function updateSignalPanel(data) {
  const pb = parseFloat(data.percent_b ?? 0.5);

  // %b gauge
  el("pb-value").textContent = fmt(pb, 4);
  const gaugeEl = el("pb-gauge");
  gaugeEl.style.width      = Math.min(Math.max(pb * 100, 0), 100) + "%";
  gaugeEl.style.background = pbColour(pb);

  // RSI / ATR
  el("rsi-value").textContent = fmt(data.rsi, 1);
  el("atr-value").textContent = fmt(data.atr, 2);

  // ML score
  const score = parseFloat(data.signal_quality ?? data.signal_quality_score ?? 0);
  el("score-value").textContent = fmt(score, 3);
  const scoreBar = el("score-bar");
  scoreBar.style.width      = (score * 100) + "%";
  scoreBar.style.background = scoreColour(score);

  // Signal badge
  const sig = (data.signal || "NONE").toUpperCase();
  const sigClass = sig === "CE" ? "green" : sig === "PE" ? "red" : "muted";
  applyBadge("signal-badge", sig, sigClass);

  // Regime
  const rname = (data.regime_name || "UNKNOWN").toUpperCase();
  applyBadge("badge-regime", rname, regimeBadgeClass(rname));
}

// ── Active trade panel ────────────────────────────────────────────────────────

function updateActiveTrade(trade, currentPrice) {
  if (!trade || !trade.is_open) {
    applyBadge("active-trade-badge", "NONE", "muted");
    el("active-pnl").textContent = "₹ —";
    el("active-pnl").className = "active-trade-pnl";
    el("at-symbol").textContent = "—";
    el("at-strike").textContent = "—";
    el("at-entry").textContent  = "—";
    el("at-pb").textContent     = "—";
    el("at-score").textContent  = "—";
    return;
  }

  // Estimate live P&L using entry price vs current underlying
  const dir = trade.direction === "CE" ? 1 : -1;
  const delta = 0.45;
  const priceDiff = (currentPrice - 0) * 0; // option LTP not available here
  const unrealised = trade.pnl || 0; // use last known pnl as proxy

  const pnlEl = el("active-pnl");
  pnlEl.textContent = fmtInr(unrealised);
  pnlEl.className = "active-trade-pnl " + (unrealised >= 0 ? "pos" : "neg");

  const dirClass = trade.direction === "CE" ? "green" : "red";
  applyBadge("active-trade-badge", trade.direction, dirClass);
  el("at-symbol").textContent = trade.symbol || "—";
  el("at-strike").textContent = trade.strike || "—";
  el("at-entry").textContent  = fmt(trade.entry_price, 2);
  el("at-pb").textContent     = fmt(trade.entry_pb, 3);
  el("at-score").textContent  = fmt(trade.signal_quality_score, 2);
}

// ── Today's trades table ──────────────────────────────────────────────────────

function updateTradesTable(trades) {
  const tbody = el("trades-tbody");
  el("today-trade-count").textContent = trades.length + " trade" + (trades.length === 1 ? "" : "s");

  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="no-data">No trades today</td></tr>';
    return;
  }

  tbody.innerHTML = trades.map(t => {
    const pnl = parseFloat(t.pnl || 0);
    const rowClass = pnl >= 0 ? "win" : "loss";
    const pnlClass = pnl >= 0 ? "pnl-pos" : "pnl-neg";
    const entryTime = t.entry_time ? t.entry_time.split("T")[1]?.substring(0, 8) : "—";
    return `<tr class="${rowClass}">
      <td>${entryTime}</td>
      <td><span class="badge ${t.direction==='CE'?'green':'red'}" style="padding:1px 6px;">${t.direction||"—"}</span></td>
      <td class="text-mono" style="font-size:10px;">${(t.symbol||"—").substring(0,16)}</td>
      <td>${fmt(t.entry_price,2)}</td>
      <td>${fmt(t.exit_price,2)}</td>
      <td class="${pnlClass}">${fmtInr(pnl)}</td>
      <td style="font-size:10px;">${t.exit_reason||"—"}</td>
      <td>${fmt(t.signal_quality_score,2)}</td>
    </tr>`;
  }).join("");
}

// ── Controls ──────────────────────────────────────────────────────────────────

function updateControls(status) {
  el("ctrl-pnl").textContent    = fmtInr(status.daily_pnl || 0);
  el("ctrl-trades").textContent = status.trades_today || 0;
}

// ── TradingView Lightweight Chart ─────────────────────────────────────────────

function initChart() {
  const container = el("tv-chart");
  tvChart = LightweightCharts.createChart(container, {
    layout:     { background: { color: "#0d1117" }, textColor: "#8b949e" },
    grid:       { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
    crosshair:  { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: "#30363d" },
    timeScale:  { borderColor: "#30363d", timeVisible: true, secondsVisible: false },
    width:  container.clientWidth,
    height: container.clientHeight,
  });

  candleSeries = tvChart.addCandlestickSeries({
    upColor:   "#3fb950", downColor: "#f85149",
    borderUpColor: "#3fb950", borderDownColor: "#f85149",
    wickUpColor: "#3fb950", wickDownColor: "#f85149",
  });

  bbUpperSeries = tvChart.addLineSeries({ color: "rgba(88,166,255,0.5)", lineWidth: 1, title: "BB Upper" });
  bbMidSeries   = tvChart.addLineSeries({ color: "rgba(88,166,255,0.8)", lineWidth: 1, lineStyle: 2, title: "BB Mid" });
  bbLowSeries   = tvChart.addLineSeries({ color: "rgba(88,166,255,0.5)", lineWidth: 1, title: "BB Lower" });

  // Resize observer
  new ResizeObserver(() => {
    tvChart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  }).observe(container);
}

function pushCandle(candle) {
  if (!candleSeries) return;
  // candle: { time, open, high, low, close, bb_upper, bb_middle, bb_lower }
  const t = typeof candle.time === "number" ? candle.time : tsToUnix(candle.time);
  candleSeries.update({ time: t, open: candle.open, high: candle.high, low: candle.low, close: candle.close });
  if (candle.bb_upper)  bbUpperSeries.update({ time: t, value: candle.bb_upper });
  if (candle.bb_middle) bbMidSeries.update({   time: t, value: candle.bb_middle });
  if (candle.bb_lower)  bbLowSeries.update({   time: t, value: candle.bb_lower });
}

function addTradeMarker(time, direction, markerType) {
  if (!candleSeries) return;
  // markerType: "entry" | "exit" | "sl"
  const t = typeof time === "number" ? time : tsToUnix(time);
  const isEntry = markerType === "entry";
  const isSL    = markerType === "sl";
  candleSeries.setMarkers([
    ...((candleSeries.markers && candleSeries.markers()) || []),
    {
      time:     t,
      position: direction === "CE" ? "belowBar" : "aboveBar",
      color:    isEntry ? "#3fb950" : (isSL ? "#f85149" : "#d29922"),
      shape:    isEntry ? "arrowUp"  : (isSL ? "circle"  : "arrowDown"),
      text:     isEntry ? `▲ ${direction}` : (isSL ? "✕ SL" : "▼ Exit"),
    },
  ]);
}

// ── Backtest mini chart ────────────────────────────────────────────────────────

function initBacktestChart() {
  const c = el("backtest-equity-chart");
  if (!c || btEquityChart) return;
  btEquityChart = LightweightCharts.createChart(c, {
    layout:     { background: { color: "#161b22" }, textColor: "#8b949e" },
    grid:       { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
    rightPriceScale: { borderColor: "#30363d" },
    timeScale:  { borderColor: "#30363d", timeVisible: true },
    width:  c.clientWidth,
    height: 120,
    handleScroll: false,
    handleScale:  false,
  });
  btEquitySeries = btEquityChart.addAreaSeries({
    lineColor: "#388bfd",
    topColor:  "rgba(56,139,253,0.3)",
    bottomColor: "rgba(56,139,253,0.01)",
    lineWidth: 2,
  });
}

function renderBacktestResults(data) {
  const m = data.metrics || {};
  const sign = v => parseFloat(v) >= 0 ? "" : "";

  el("bt-sharpe").textContent  = fmt(m.sharpe, 2);
  el("bt-winrate").textContent = fmtPct(m.win_rate);
  el("bt-pnl").textContent     = fmtInr(m.total_pnl);
  el("bt-dd").textContent      = fmtInr(m.max_drawdown_inr);
  el("bt-trades").textContent  = m.total_trades || "—";
  el("bt-tw").textContent      = fmt(m.trades_per_week, 1);

  el("bt-pnl").className = "value " + (parseFloat(m.total_pnl) >= 0 ? "pos" : "neg");
  el("bt-sharpe").className = "value " + (parseFloat(m.sharpe) >= 0 ? "pos" : "neg");

  // Build equity curve from trades
  if (data.trades && data.trades.length > 0 && btEquitySeries) {
    let cumPnl = 0;
    const points = data.trades
      .filter(t => t.exit_time)
      .map(t => {
        cumPnl += parseFloat(t.pnl || 0);
        return { time: tsToUnix(t.exit_time), value: parseFloat(cumPnl.toFixed(2)) };
      })
      .sort((a, b) => a.time - b.time);

    // Deduplicate times
    const seen = new Set();
    const deduped = points.filter(p => {
      if (seen.has(p.time)) return false;
      seen.add(p.time);
      return true;
    });

    if (deduped.length > 0) btEquitySeries.setData(deduped);
  }
}

// ── Model status ──────────────────────────────────────────────────────────────

function updateModelStatus(data) {
  const regime = data.regime_model || {};
  const filter = data.signal_filter_model || {};
  const params = data.optimized_params || {};
  const meta   = params.meta || {};

  el("ms-regime").textContent     = regime.exists  ? "✓ Loaded" : "✗ Missing";
  el("ms-regime").style.color     = regime.exists  ? "var(--green)" : "var(--red)";
  el("ms-filter").textContent     = filter.exists  ? "✓ Loaded" : "✗ Missing";
  el("ms-filter").style.color     = filter.exists  ? "var(--green)" : "var(--red)";

  const paramsDate = params.last_modified || "—";
  el("ms-params-date").textContent = paramsDate !== "N/A"
    ? paramsDate.split("T")[0] : "—";

  const oos = meta?.walk_forward_summary?.avg_oos_sharpe;
  el("ms-sharpe").textContent = oos != null ? fmt(oos, 2) : "—";

  const v = params.values || {};
  el("p-oversold").textContent  = v.bb_oversold  != null ? fmt(v.bb_oversold, 3)  : "—";
  el("p-overbought").textContent = v.bb_overbought != null ? fmt(v.bb_overbought, 3) : "—";
  el("p-exit").textContent      = v.bb_exit       != null ? fmt(v.bb_exit, 3)      : "—";
  el("p-sl").textContent        = v.sl_buffer     != null ? fmt(v.sl_buffer, 3)    : "—";
  el("p-rsi-min").textContent   = v.rsi_min       != null ? v.rsi_min              : "—";
  el("p-rsi-max").textContent   = v.rsi_max       != null ? v.rsi_max              : "—";
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWS() {
  if (ws && ws.readyState <= 1) return;

  ws = new WebSocket(wsUrl());

  ws.onopen = () => {
    el("conn-dot").className   = "connected";
    el("conn-label").textContent = "WS ●";
    if (wsRetryTimer) { clearTimeout(wsRetryTimer); wsRetryTimer = null; }
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateHeader(data);
      updateSignalPanel(data);
      updateActiveTrade(data.active_trade, data.price);

      // If there's a live price, push a synthetic candle update
      if (data.price && candleSeries) {
        const now = Math.floor(Date.now() / 1000);
        candleSeries.update({ time: now, open: data.price, high: data.price, low: data.price, close: data.price });
      }
    } catch (e) {
      console.warn("WS parse error", e);
    }
  };

  ws.onerror  = () => { el("conn-dot").className = "error"; };
  ws.onclose  = () => {
    el("conn-dot").className   = "error";
    el("conn-label").textContent = "WS ✗";
    wsRetryTimer = setTimeout(connectWS, 5000);
  };
}

// ── REST polling ──────────────────────────────────────────────────────────────

async function fetchJSON(path) {
  const resp = await fetch(BACKEND_URL + path, { headers: authHeaders() });
  if (resp.status === 401 || resp.status === 503) {
    localStorage.removeItem(API_TOKEN_STORAGE_KEY);
    apiTokenPrompted = false;
  }
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

async function postJSON(path, body) {
  const resp = await fetch(BACKEND_URL + path, {
    method:  "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body:    body ? JSON.stringify(body) : undefined,
  });
  if (resp.status === 401 || resp.status === 503) {
    localStorage.removeItem(API_TOKEN_STORAGE_KEY);
    apiTokenPrompted = false;
  }
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

async function loadCandles() {
  if (!candleSeries) return;
  try {
    const d = await fetchJSON("/candles");
    const candles = (d && d.candles) || [];
    if (!candles.length) return;
    candleSeries.setData(candles.map(c => ({
      time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
    })));
    const bbU = candles.filter(c => c.bb_upper  != null).map(c => ({ time: c.time, value: c.bb_upper  }));
    const bbM = candles.filter(c => c.bb_middle != null).map(c => ({ time: c.time, value: c.bb_middle }));
    const bbL = candles.filter(c => c.bb_lower  != null).map(c => ({ time: c.time, value: c.bb_lower  }));
    if (bbU.length) bbUpperSeries.setData(bbU);
    if (bbM.length) bbMidSeries.setData(bbM);
    if (bbL.length) bbLowSeries.setData(bbL);
    tvChart.timeScale().fitContent();
  } catch (e) { /* not authenticated yet, or market data unavailable */ }
}

async function pollStatus() {
  try {
    const s = await fetchJSON("/status");
    updateBotBadge(s.bot_running, s.market_open);
    updateControls(s);
  } catch (e) { /* ignore — WS gives the same info */ }
}

async function pollTrades() {
  try {
    const trades = await fetchJSON("/trades");
    updateTradesTable(trades);
  } catch (e) { }
}

async function loadModelStatus() {
  try {
    const d = await fetchJSON("/model/status");
    updateModelStatus(d);
  } catch (e) { el("ms-regime").textContent = "Error"; }
}

// ── Event handlers ────────────────────────────────────────────────────────────

el("btn-start").addEventListener("click", async () => {
  try {
    const r = await postJSON("/bot/start");
    toast("Bot started: " + (r.dry_run ? "DRY RUN" : "LIVE"), "ok");
  } catch (e) { toast("Failed to start bot: " + e.message, "error"); }
});

el("btn-stop").addEventListener("click", async () => {
  try {
    await postJSON("/bot/stop");
    toast("Bot stopped", "ok");
  } catch (e) { toast("Failed to stop bot: " + e.message, "error"); }
});

el("btn-run-bt").addEventListener("click", async () => {
  el("bt-status").textContent = "Running backtest…";
  try {
    initBacktestChart();
    const data = await fetchJSON("/backtest/run");
    renderBacktestResults(data);
    el("bt-status").textContent = "Done ✓";
  } catch (e) {
    el("bt-status").textContent = "Error: " + e.message;
    toast("Backtest failed: " + e.message, "error");
  }
});

el("btn-reload-model").addEventListener("click", () => { loadModelStatus(); toast("Model status refreshed"); });

el("btn-get-url").addEventListener("click", async () => {
  try {
    const d = await fetchJSON("/auth/login-url");
    const link = el("kite-login-link");
    link.href = d.login_url;
    link.style.display = "block";
    link.textContent   = "Open Kite Login →";
    toast("Login URL ready — click the link");
  } catch (e) { toast("Failed to get login URL: " + e.message, "error"); }
});

el("btn-submit-token").addEventListener("click", async () => {
  const token = el("request-token-input").value.trim();
  if (!token) { toast("Enter a request_token first", "error"); return; }
  try {
    const resp = await fetch(BACKEND_URL + "/auth/login", {
      method:  "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body:    JSON.stringify({ request_token: token }),
    });
    if (resp.status === 401 || resp.status === 503) {
      localStorage.removeItem(API_TOKEN_STORAGE_KEY);
      apiTokenPrompted = false;
    }
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || resp.status);
    }
    toast("Kite authenticated successfully!", "ok");
    el("request-token-input").value = "";
    loadCandles();
  } catch (e) { toast("Auth failed: " + e.message, "error"); }
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────

function bootstrap() {
  initChart();
  loadCandles();
  connectWS();
  loadModelStatus();
  pollStatus();
  pollTrades();

  // Poll REST every 5 seconds
  setInterval(pollStatus, 5000);
  setInterval(pollTrades, 5000);
  // Refresh model status every 5 minutes
  setInterval(loadModelStatus, 300_000);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap);
} else {
  bootstrap();
}
