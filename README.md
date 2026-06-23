# BTC Impulse AI Dashboard

A modern, real-time dashboard for analyzing BTC price action with SNR, SMC, and AI-based predictions. Built with Streamlit and ready to deploy on Railway!

## Features

- 📈 Live candlestick charts with SNR bands
- 📊 Interactive price action and market structure visualizations
- 🎯 Real-time technical indicators (RSI, ATR, Flow)
- 📤 One-click TradingView bridge export
- 🚀 Ready for Railway deployment


## What Is Inside

- `BTC_Impulse_Research.ipynb`
  Full research notebook with data loading, feature engineering, charts, target labeling, model training, evaluation, and export steps.
- `snr_ml_features.py`
  Python feature pipeline that mirrors the main ideas from `SNR_Line.pine`.
- `model_runtime.py`
  Training, saving, loading, and inference helpers for the baseline AI models.
- `integration_bridge.py`
  Utility module for exporting model outputs into a lightweight signal format that can later be consumed by dashboards, Pine-side workflows, or external services.
- `screenshot_reader.py`
  Screenshot fallback module that extracts TradingView context through image + OCR parsing.
- `run_pipeline.py`
  Product runner that loads data, trains or loads models, exports signals, and can trigger Telegram delivery.
- `run_train.py`
  Separate training entry point for re-training and saving the model bundle.
- `run_infer.py`
  Separate inference entry point for exporting the latest signal without retraining.
- `product_config.example.json`
  Example runtime configuration for the full product pipeline.
- `telegram_bot_stub.py`
  Starter module for future Telegram notifications.
- `requirements.txt`
  Suggested Python dependencies.

## Research Goal

Predict the probability of a large BTC impulse, with special focus on:

- New York session
- Kill Zones
- Silver Bullet
- SNR reactions
- BOS / CHoCH / FVG context
- Price action confirmation
- Screenshot fallback when fresh structured data is unavailable

## Suggested Workflow

1. Install Python 3.11+.
2. Create a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Open `BTC_Impulse_Research.ipynb`.
5. Load BTC data from CSV or online source.
6. Train and evaluate the first baseline model.
7. Run `run_pipeline.py` to export model outputs.
8. Use screenshot fallback from `inbox/` when fresh data is unavailable.
9. Later connect Telegram alerts through `telegram_bot_stub.py`.

## Recommended Data

- Symbol: `BTC-USD` or exchange-specific BTC/USDT
- Timeframes: `1m`, `5m`, `15m`
- Fields: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Prefer at least 3-12 months of data for initial study
- For real-time notebook work, use `data_source = "auto"` so the project tries Binance first and falls back to yfinance
- If you want a fresh Yahoo pull anchored to the current moment, use `data_source = "yfinance_now"` with `datetime.now(timezone.utc)`
- For deep backtests beyond intraday API limits, prefer a local CSV file

## Target Idea

The default notebook studies whether a large move happens in the next `N` bars, for example:

- `future_up_move >= threshold`
- `future_down_move >= threshold`
- or a 3-class target:
  - no large move
  - large up move
  - large down move

## Integration Direction

The model does not replace the indicator. It acts as a filter on top of the current logic:

- Indicator finds `SNR + KZ + FVG + BOS/CHoCH + PA`
- Model estimates impulse probability
- Exported signal can later be routed to Telegram

## Product Flow

### Mode 1: Structured Market Data

1. Load OHLCV data from CSV or `yfinance`
2. Build `SNR/SMC/KZ/PA` features
3. Train or load baseline models
4. Export latest signal into `outputs/latest_signal.json`
5. Export lightweight indicator snapshot into `outputs/indicator_snapshot.json`
6. Export ready-to-paste TradingView bridge payload into `outputs/tradingview_ai_bridge.txt`
7. Optionally send Telegram alert

#### Useful Commands

```powershell
py run_train.py product_config.example.json
```

```powershell
py run_infer.py product_config.example.json
```

```powershell
.\copy_bridge_to_clipboard.ps1
```

```powershell
.\copy_bridge_to_clipboard.ps1 -RunInfer
```

## TradingView Bridge Workflow

`TradingView / Pine` cannot read local JSON files directly, so the bridge now uses a single compact payload string.

1. Run `run_infer.py` or `run_pipeline.py`
2. Open `outputs/indicator_snapshot.json` if you want the full exported snapshot
3. Copy either:
   - `pine_bridge_payload` from `indicator_snapshot.json`
   - or the full contents of `outputs/tradingview_ai_bridge.txt`
4. In `SNR_Line.pine` open the `AI Integration` group
5. Turn on `Use AI Bridge Payload`
6. Paste the copied string into `AI Bridge Payload`
7. `AI Direction`, `AI Large Move %`, `AI Direction %`, `AI Session`, `AI Entry State`, `AI Notes`, plus `PA Signal / PA Bias / PA Reason` are then filled from that one payload automatically inside the indicator logic
8. For the fastest Windows workflow, run `.\copy_bridge_to_clipboard.ps1` to copy the latest bridge payload directly into the clipboard

Example payload:

```text
AIBRIDGE|Bullish|72.4|64.8|London KZ|SNR Long|Inference export aligned with SNR_Line.pine|Bull Breakout|Long|Price broke structure or SNR resistance
```

## Real-Time Notebook Mode

`BTC_Impulse_Research.ipynb` now supports fresh live data directly.

1. Open the notebook.
2. In the config cell keep `DATA_SOURCE = 'auto'`.
3. Run the config cell to fetch fresh market data.
4. Run the `Live Data Monitor` cell for a current snapshot.
5. Optionally call `monitor_live_data(iterations=20, sleep_seconds=30)` for repeated refreshes.
6. If you want near real-time streaming, use the notebook `Binance Websocket Mode`.
7. If you want the latest Yahoo-compatible pull, compare the `Yfinance Now Snapshot` section which uses `datetime.now(timezone.utc)`.

Notes:

- `auto` tries Binance first, which is better for fresh crypto intraday data.
- `yfinance_now` uses an explicit `start/end` window ending at `datetime.now(timezone.utc)`.
- If live APIs are unavailable, switch to CSV.
- The default live notebook period is shorter on purpose so intraday requests stay reliable.

### Binance Websocket Mode

For faster updates than polling:

1. Install `websocket-client` from `requirements.txt`
2. In the notebook run the `Binance Websocket Mode` cells
3. Wait for the first kline events
4. Optionally call `monitor_websocket_stream(ws_stream, iterations=30, sleep_seconds=2)`
5. Stop the stream with `ws_stream.stop()`

This mode is useful when you want a near real-time BTC flow in the notebook while keeping the research and bridge export in the same workspace.

### Mode 2: Screenshot Fallback

1. Put TradingView screenshot into `inbox/`
2. `screenshot_reader.py` extracts OCR text and visual hints
3. Latest screenshot snapshot is saved into `snapshots/latest_screenshot_snapshot.json`
4. The screenshot context can be used when fresh market data is unavailable or as a confirmation layer

## Folders

- `inbox/`
  Put screenshots here for fallback parsing.
- `models/`
  Saved baseline models and metrics.
- `outputs/`
  Latest AI payload and signal history.
- `snapshots/`
  Screenshot-derived context snapshots.

## Running the Dashboard Locally

1. Make sure you have Python 3.11+ installed
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
5. Open your browser and navigate to `http://localhost:8501`

## Deploying on Railway

1. Install the Railway CLI: https://docs.railway.app/guides/cli
2. Login to Railway:
   ```bash
   railway login
   ```
3. Initialize a new Railway project:
   ```bash
   railway init
   ```
4. Deploy your project:
   ```bash
   railway up
   ```
5. That's it! Your dashboard will be live on Railway.

## Notes

- This project intentionally starts with tree-based ML because it is faster to validate than deep learning.
- The notebook is written to avoid look-ahead leakage when preparing labels and features.
- Pine Script cannot read local JSON files directly, so AI integration inside TradingView is handled here through a compact external bridge payload generated from `indicator_snapshot.json`.
