# NIFTY BB Bot — Algorithmic Trading System

Production-grade NIFTY50 weekly options trading bot with ML-enhanced signal generation,
real-time dashboard, and automated execution via Zerodha Kite Connect.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Pages                                  │
│   frontend/  ──  index.html + app.js + style.css                    │
│   TradingView chart │ Signal panel │ Trade table │ Backtest panel   │
└────────────────────────────┬────────────────────────────────────────┘
                             │  REST + WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Railway (FastAPI)                              │
│                                                                     │
│  api_server.py  ──  /status /indicators /trades /backtest /ws/live │
│  main.py        ──  5-min candle pipeline + asyncio bot loop        │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────────┐  │
│  │  data_feed   │  │   ml/ layer      │  │   backtester/       │  │
│  │  WebSocket   │  │  ┌────────────┐  │  │  engine.py          │  │
│  │  → Candles   │  │  │  regime    │  │  │  metrics.py         │  │
│  └──────────────┘  │  │  detector  │  │  │  walk_forward.py    │  │
│                    │  │  (HMM)     │  │  └─────────────────────┘  │
│  ┌──────────────┐  │  ├────────────┤  │                           │
│  │  options_    │  │  │  signal    │  │  ┌─────────────────────┐  │
│  │  selector   │  │  │  filter    │  │  │  order_manager.py   │  │
│  └──────────────┘  │  │  (XGB)     │  │  │  CSV trade log      │  │
│                    │  ├────────────┤  │  └─────────────────────┘  │
│                    │  │  param     │  │                           │
│                    │  │  optimizer │  │                           │
│                    │  │  (Optuna)  │  │                           │
│                    │  └────────────┘  │                           │
│                    └──────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │   Zerodha Kite Connect   │
              │   NSE/NFO order routing  │
              └──────────────────────────┘
```

---

## 2. Quickstart

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/nifty-bb-bot.git
cd nifty-bb-bot

# 2. Fill in environment variables
cp backend/.env.example backend/.env
# Edit backend/.env with your Kite API keys, API_AUTH_TOKEN, etc.

# 3. Install Python dependencies
cd backend
pip install -r requirements.txt

# 4. Train ML models (first run)
python ml/train.py --months 9 --trials 200

# 5. Start the backend server
python main.py
```

The API server starts on `http://localhost:8000`.

---

## 3. Kite Login (First Run)

The bot uses Zerodha Kite Connect OAuth. On first run:

1. Open the dashboard at your GitHub Pages URL (or `frontend/index.html` locally)
2. Click **"Get Login URL"** in the Kite Authentication panel
3. A Kite login link appears — click it to open Kite's login page in your browser
4. Log in with your Zerodha credentials
5. After login, you will be redirected to a URL containing `?request_token=XXXXXXXX`
6. Copy the `request_token` value from that URL
7. Paste it into the **"Paste request_token"** input and click **"Submit Token"**

The access token is now stored in the server process for the trading day.

> **Note:** Kite access tokens expire daily. You must re-authenticate each trading morning.
> Consider automating this by storing the token in a database or using Kite's token refresh mechanism.

---

## 4. Training Models

```bash
cd backend

# Full training run (9 months data, 200 Bayesian trials)
python ml/train.py --months 9 --trials 200

# Quick smoke-test with synthetic data (no Kite needed)
python ml/train.py --months 3 --trials 20 --demo

# Output artifacts:
#   ml/models/regime_model.joblib         ← HMM regime classifier
#   ml/models/signal_filter_model.joblib  ← XGBoost signal filter
#   ml/models/optimized_params.json       ← Bayesian-optimized parameters
```

---

## 5. Deploy Backend to Railway

1. Create a new Railway project at [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Set the root directory to `backend/`
4. Railway will automatically detect the `Procfile`:
   ```
   web: uvicorn api_server:app --host 0.0.0.0 --port $PORT
   ```
5. Add environment variables in the Railway dashboard (copy from `.env.example`)
6. Deploy — Railway gives you a public URL like `https://your-app.railway.app`

---

## 6. Enable GitHub Pages (Frontend)

1. Push the repository to GitHub
2. Go to **Settings → Pages** in your repo
3. Source: **Deploy from a branch** → Branch: `gh-pages` → Root `/`
4. The frontend auto-deploys on every push to `main` via `.github/workflows/deploy_frontend.yml`
5. Update `BACKEND_URL` in `frontend/app.js` to your Railway URL before pushing

---

## 7. Weekly Retraining (GitHub Actions)

The workflow `.github/workflows/retrain.yml` runs automatically every **Friday at 6 PM IST**.

It:
1. Fetches the last 9 months of NIFTY 5-min data from Kite
2. Re-trains the HMM regime detector and XGBoost signal filter
3. Runs 200 Bayesian optimization trials for strategy parameters
4. Commits the updated model artifacts back to the repo
5. The frontend re-deploys automatically picking up new parameters

**Required GitHub Secrets** (Settings → Secrets → Actions):
```
KITE_API_KEY
KITE_API_SECRET
KITE_ACCESS_TOKEN    ← refresh this every time the token expires
```

You can also trigger a manual retraining from the **Actions** tab → "Weekly Model Retraining" → "Run workflow".

---

## 8. DRY RUN Mode

When `DRY_RUN=true` (default in `.env.example`):
- All orders are **logged** but `kite.place_order()` is **never called**
- Log lines include a `[DRY RUN]` prefix
- All P&L figures are simulated based on LTP at signal time
- The trade log CSV is still written

To go live: set `DRY_RUN=false` in your environment.

---

## 9. Trade Log

Trades are persisted to `backend/trades.csv`. Schema:

| Column               | Description                                    |
|----------------------|------------------------------------------------|
| trade_id             | Unique ID e.g. T20240612-001                   |
| entry_time           | ISO timestamp of entry                         |
| exit_time            | ISO timestamp of exit                          |
| direction            | CE or PE                                       |
| symbol               | Kite tradingsymbol e.g. NIFTY2461222000CE      |
| strike               | Strike price                                   |
| entry_price          | Option LTP at entry (with slippage)            |
| exit_price           | Option LTP at exit (with slippage)             |
| quantity             | Number of shares (lots × 25)                   |
| pnl                  | Net P&L after brokerage                        |
| exit_reason          | TARGET / STOP_LOSS / FORCE_EXIT / BOT_STOP     |
| signal_quality_score | XGBoost quality score at entry (0–1)           |
| entry_pb             | Bollinger %b at entry                          |
| exit_pb              | Bollinger %b at exit                           |
| regime               | Market regime at entry (0=Down,1=Choppy,2=Up)  |

---

## 10. Reading the Dashboard

| Panel              | What to look for                                               |
|--------------------|----------------------------------------------------------------|
| **Header**         | Live NIFTY price, Market status, Bot on/off, Regime badge      |
| **Signal panel**   | %b gauge: red zones (< 0.1 or > 0.9) = potential signals      |
|                    | ML Score bar: must be ≥ 0.60 for a trade to fire              |
|                    | Regime badge: GREEN = CHOPPY (trades allowed), YELLOW = filtered|
| **Chart**          | 5-min NIFTY candles with Bollinger Band overlay               |
|                    | ▲ green = entry, ▼ yellow = exit, ✕ red = stop-loss           |
| **Model Status**   | Check "Params Updated" date — should be recent (≤ 7 days)     |
|                    | OOS Sharpe should be > 0.5 for the bot to be worth running    |
| **Active Trade**   | Unrealised P&L updates in real-time via WebSocket             |
| **Today's Trades** | Color-coded: green row = winner, red row = loser              |
| **Backtest**       | Click "Run" to back-test last 90 days with current parameters |
|                    | Equity curve in the mini chart below the metrics              |
| **Bot Controls**   | DRY RUN toggle (enable before going live)                     |
|                    | START / STOP buttons call the FastAPI bot control endpoints   |

---

## Strategy Logic (Quick Reference)

```
Every 5-min candle close:
  1. Compute indicators (BB %b, RSI, ATR, EMA)
  2. HMM regime check → skip if NOT CHOPPY
  3. Check %b signal:
       %b < bb_oversold  → potential CE entry
       %b > bb_overbought → potential PE entry
  4. XGBoost quality score → skip if score < 0.60
  5. Place ATM option order (if within MAX_TRADES_PER_DAY=3 limit)

  Exit conditions (checked each candle while in trade):
    %b crosses bb_exit   → TARGET exit
    %b crosses SL level  → STOP_LOSS exit
    15:10 IST            → FORCE_EXIT (square off)
```

---

## Environment Variables Reference

```env
KITE_API_KEY=           # From Zerodha developer console
KITE_API_SECRET=        # From Zerodha developer console
KITE_ACCESS_TOKEN=      # Generated daily via OAuth flow
FRONTEND_ORIGIN=        # e.g. https://yourname.github.io
API_AUTH_TOKEN=         # Long random token required by the dashboard/API
PORT=8000               # Server port (Railway sets this automatically)
DRY_RUN=true            # Set false only when ready for live trading
LOG_LEVEL=INFO          # DEBUG / INFO / WARNING / ERROR
```
