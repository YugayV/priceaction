
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json

from snr_ml_features import FeatureConfig, add_snr_features, build_impulse_targets, load_market_data
from integration_bridge import build_signal_payload, save_indicator_snapshot, save_tradingview_bridge
from news_analyzer import NewsAnalyzer
from deepseek_analyzer import DeepSeekAnalyzer
from model_runtime import train_model_bundle, predict_latest

st.set_page_config(
    page_title="Market Intelligence AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Assets Config
with open("assets_config.json", "r", encoding="utf-8") as f:
    assets_config = json.load(f)
assets = assets_config["assets"]

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

asset_key = st.sidebar.selectbox(
    "Select Asset",
    list(assets.keys()),
    index=0,
    format_func=lambda k: assets[k]["name"]
)

symbol = assets[asset_key]["symbol"]
interval = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h"], index=2)
period = st.sidebar.selectbox("Data Period", ["30d", "60d", "120d", "180d", "365d"], index=2)
timezone = st.sidebar.text_input("Timezone", "Asia/Seoul")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Features")
enable_deepseek = st.sidebar.checkbox("Enable DeepSeek Analysis", value=False)
enable_news = st.sidebar.checkbox("Enable News Analysis", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 ML Model")
ml_horizon = st.sidebar.slider("Target horizon (bars)", min_value=4, max_value=48, value=12, step=1)
ml_threshold_pct = st.sidebar.slider("Large move threshold (%)", min_value=0.10, max_value=5.00, value=0.80, step=0.05)
ml_split_ratio = st.sidebar.slider("Train/Test split", min_value=0.60, max_value=0.95, value=0.80, step=0.05)
train_models = st.sidebar.button("Train ML models")

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Refresh Data")
refresh_data = st.sidebar.button("Refresh All Data")

# Main Dashboard
st.title(f"📊 {assets[asset_key]['name']} Market Intelligence Dashboard")
st.markdown("""
Комплексная аналитика рынка с CVision, ML моделями и DeepSeek AI.
""")

# Load Data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_data(symbol, interval, period, timezone):
    cfg = FeatureConfig(
        symbol=symbol,
        interval=interval,
        period=period,
        timezone=timezone,
    )
    return load_market_data(
        symbol=cfg.symbol,
        interval=cfg.interval,
        period=cfg.period,
        timezone=cfg.timezone,
        data_source="auto",
    )

@st.cache_data(ttl=300)
def get_features(data):
    return add_snr_features(data)

@st.cache_data(ttl=600)
def get_news(asset_key):
    news_analyzer = NewsAnalyzer()
    return news_analyzer.get_news_for_asset(asset_key)

def _recommendation(df_row: pd.Series, probs: dict, horizon_bars: int, threshold_pct: float) -> dict:
    event_prob = float(probs.get("large_move_probability", 0.0))
    up_prob = float(probs.get("up_move_probability", 0.5))
    down_prob = float(probs.get("down_move_probability", 0.5))

    long_bias = (
        int(df_row.get("trend_bull", 0)) == 1
        and int(df_row.get("flow_is_bull", 0)) == 1
        and float(df_row.get("kz_long_score", 0.0)) >= float(df_row.get("kz_short_score", 0.0))
    )
    short_bias = (
        int(df_row.get("trend_bear", 0)) == 1
        and int(df_row.get("flow_is_bull", 1)) == 0
        and float(df_row.get("kz_short_score", 0.0)) >= float(df_row.get("kz_long_score", 0.0))
    )

    direction = "NEUTRAL"
    if event_prob >= 0.65 and up_prob >= 0.60 and long_bias:
        direction = "LONG"
    elif event_prob >= 0.65 and down_prob >= 0.60 and short_bias:
        direction = "SHORT"

    close = float(df_row["close"])
    atr = float(df_row.get("atr14", np.nan))
    basis = float(df_row.get("reg_basis", close))
    snr_upper = float(df_row.get("snr_upper", close))
    snr_lower = float(df_row.get("snr_lower", close))

    entry = basis
    if direction == "LONG":
        entry = max(min(close, basis), snr_lower)
    if direction == "SHORT":
        entry = min(max(close, basis), snr_upper)

    stop = None
    take = None
    if np.isfinite(atr) and atr > 0:
        if direction == "LONG":
            stop = entry - atr * 1.6
            take = entry + atr * 3.0
        elif direction == "SHORT":
            stop = entry + atr * 1.6
            take = entry - atr * 3.0

    return {
        "direction": direction,
        "event_prob": event_prob,
        "up_prob": up_prob,
        "down_prob": down_prob,
        "entry": float(entry),
        "stop": float(stop) if stop is not None else None,
        "take": float(take) if take is not None else None,
        "horizon_bars": int(horizon_bars),
        "threshold_pct": float(threshold_pct),
        "timestamp": str(df_row.name),
    }

def _build_dashboard_figure(df: pd.DataFrame, show_snr: bool, show_ema: bool, show_bb: bool, show_vwap: bool, show_signals: bool) -> go.Figure:
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.52, 0.12, 0.12, 0.12, 0.12],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}], [{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    if show_snr:
        fig.add_trace(go.Scatter(x=df.index, y=df["reg_basis"], mode="lines", name="SNR Basis", line=dict(color="#00ff88", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["snr_upper"], mode="lines", name="SNR Upper", line=dict(color="#ff4444", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["snr_lower"], mode="lines", name="SNR Lower", line=dict(color="#44ff44", width=1, dash="dash")), row=1, col=1)

    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], mode="lines", name="EMA 20", line=dict(color="#ffd166", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], mode="lines", name="EMA 50", line=dict(color="#118ab2", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema200"], mode="lines", name="EMA 200", line=dict(color="#6a4c93", width=1)), row=1, col=1)

    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap_d"], mode="lines", name="VWAP (Daily)", line=dict(color="#ef476f", width=1)), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], mode="lines", name="BB Upper", line=dict(color="#9b5de5", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_mid"], mode="lines", name="BB Mid", line=dict(color="#9b5de5", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], mode="lines", name="BB Lower", line=dict(color="#9b5de5", width=1, dash="dot")), row=1, col=1)

    if show_signals:
        bull_idx = df.index[df["bull_sweep"] == 1] if "bull_sweep" in df.columns else df.index[:0]
        bear_idx = df.index[df["bear_sweep"] == 1] if "bear_sweep" in df.columns else df.index[:0]
        if len(bull_idx) > 0:
            fig.add_trace(go.Scatter(x=bull_idx, y=df.loc[bull_idx, "close"], mode="markers", name="Bull Sweep", marker=dict(size=8, color="#06d6a0")), row=1, col=1)
        if len(bear_idx) > 0:
            fig.add_trace(go.Scatter(x=bear_idx, y=df.loc[bear_idx, "close"], mode="markers", name="Bear Sweep", marker=dict(size=8, color="#ef476f")), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color="#3a86ff"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], mode="lines", name="RSI", line=dict(color="#ff006e", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], mode="lines", name="MACD", line=dict(color="#00b4d8", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], mode="lines", name="MACD Signal", line=dict(color="#ffd166", width=1.0)), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="MACD Hist", marker_color="#90be6d"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["atr14"], mode="lines", name="ATR", line=dict(color="#c77dff", width=1.3)), row=5, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=980,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_yaxes(title_text="ATR", row=5, col=1)
    return fig

try:
    data_load_state = st.text("Loading market data...")
    df = get_market_data(symbol, interval, period, timezone)
    data_load_state.text("")
    
    if refresh_data:
        st.cache_data.clear()
        df = get_market_data(symbol, interval, period, timezone)
    
    df_features = get_features(df)
    df_features = build_impulse_targets(df_features, horizon=int(ml_horizon), percent_threshold=float(ml_threshold_pct))
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    latest_close = df["close"].iloc[-1]
    latest_vol = df["volume"].iloc[-1]
    price_change = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100
    vol_change = (df["volume"].iloc[-1] - df["volume"].iloc[-2]) / df["volume"].iloc[-2] * 100
    
    with col1:
        st.metric(
            label="Close Price",
            value=f"${latest_close:,.2f}",
            delta=f"{price_change:+.2f}%",
        )
    with col2:
        st.metric(
            label="Volume",
            value=f"{latest_vol:,.0f}",
            delta=f"{vol_change:+.2f}%",
        )
    with col3:
        st.metric(
            label="RSI (14)",
            value=f"{df_features['rsi'].iloc[-1]:.1f}",
        )
    with col4:
        st.metric(
            label="ATR (14)",
            value=f"{df_features['atr14'].iloc[-1]:.2f}",
        )
    
    st.markdown("---")
    
    # Main Tabs
    main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs([
        "📈 Графики",
        "📟 Индикаторы",
        "🧠 ML",
        "📰 Новости",
        "🤖 DeepSeek",
        "📤 Export"
    ])
    
    # Price Action Tab
    with main_tab1:
        st.subheader("Мульти-график: цена + индикаторы")
        show_snr = st.checkbox("SNR bands", value=True)
        show_ema = st.checkbox("EMA 20/50/200", value=True)
        show_bb = st.checkbox("Bollinger Bands", value=False)
        show_vwap = st.checkbox("VWAP (Daily)", value=True)
        show_signals = st.checkbox("PA сигналы (sweep)", value=True)

        chart_df = df_features.dropna(subset=["open", "high", "low", "close"]).tail(1200).copy()
        fig = _build_dashboard_figure(chart_df, show_snr=show_snr, show_ema=show_ema, show_bb=show_bb, show_vwap=show_vwap, show_signals=show_signals)
        st.plotly_chart(fig, use_container_width=True)

    with main_tab2:
        st.subheader("Таблица индикаторов (последние 50 свечей)")
        show_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "atr14",
            "ema20",
            "ema50",
            "ema200",
            "vwap_d",
            "bb_mid",
            "bb_upper",
            "bb_lower",
            "macd",
            "macd_signal",
            "macd_hist",
            "stoch_k",
            "stoch_d",
            "flow_main",
            "kz_long_score",
            "kz_short_score",
            "bull_choch",
            "bear_choch",
            "bull_sweep",
            "bear_sweep",
        ]
        existing_cols = [c for c in show_cols if c in df_features.columns]
        st.dataframe(df_features[existing_cols].tail(50), use_container_width=True)

    with main_tab3:
        st.subheader("Машинное обучение: вероятность импульса и направление")

        if train_models or ("model_bundle" not in st.session_state):
            with st.spinner("Обучаю модели..."):
                bundle, metrics = train_model_bundle(
                    df_features,
                    split_ratio=float(ml_split_ratio),
                )
                st.session_state["model_bundle"] = bundle
                st.session_state["model_metrics"] = metrics

        bundle = st.session_state.get("model_bundle")
        metrics = st.session_state.get("model_metrics", {})

        if bundle is not None:
            probs = predict_latest(bundle, df_features)
            last_row = df_features.dropna().iloc[-1]
            rec = _recommendation(last_row, probs, horizon_bars=int(ml_horizon), threshold_pct=float(ml_threshold_pct))

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("P(impulse)", f"{rec['event_prob']*100:.1f}%")
            with c2:
                st.metric("P(up | impulse)", f"{rec['up_prob']*100:.1f}%")
            with c3:
                st.metric("Рекомендация", rec["direction"])

            st.markdown("### План сделки (ориентир)")
            st.json({k: rec[k] for k in ["direction", "entry", "stop", "take", "horizon_bars", "threshold_pct", "timestamp"]})

            st.markdown("### Метрики обучения")
            st.json({k: v for k, v in metrics.items() if k not in {"event_report", "direction_report"}})
            if metrics.get("event_report"):
                st.text("Event model report:\n" + str(metrics.get("event_report")))
            if metrics.get("direction_report"):
                st.text("Direction model report:\n" + str(metrics.get("direction_report")))
        else:
            st.info("Нажми Train ML models в сайдбаре.")
    
    # News Analysis Tab
    with main_tab4:
        if enable_news:
            st.subheader("Latest Market News")
            news_data = get_news(asset_key)
            
            if news_data:
                for news in news_data:
                    with st.expander(f"📰 {news['title']}"):
                        st.write(f"Source: {news['source']}")
                        st.write(f"URL: {news['url']}")
                        if news.get("summary"):
                            st.markdown("**Summary:**")
                            st.write(news["summary"])
                        if news.get("keywords"):
                            st.markdown("**Keywords:**")
                            st.write(", ".join(news["keywords"]))
            else:
                st.info("No news found at the moment.")
        else:
            st.info("News analysis is disabled. Enable it in the sidebar.")

    with main_tab5:
        if enable_deepseek:
            st.subheader("🤖 DeepSeek AI Analysis")
            if st.button("Generate Analysis"):
                with st.spinner("Analyzing market with DeepSeek AI..."):
                    try:
                        price_data = {
                            "symbol": symbol,
                            "latest_close": float(latest_close),
                            "price_change": float(price_change),
                            "volume": float(latest_vol),
                            "rsi": float(df_features["rsi"].iloc[-1]),
                            "atr": float(df_features["atr14"].iloc[-1]),
                            "macd": float(df_features["macd"].iloc[-1]) if "macd" in df_features.columns else None,
                            "vwap_d": float(df_features["vwap_d"].iloc[-1]) if "vwap_d" in df_features.columns else None,
                        }

                        news_data = get_news(asset_key) if enable_news else []
                        deepseek = DeepSeekAnalyzer()
                        analysis = deepseek.analyze_market(asset_key, price_data, news_data)
                        st.markdown(analysis)
                    except Exception as e:
                        st.error(f"Error during DeepSeek analysis: {str(e)}")
                        st.info("Проверь DEEPSEEK_API_KEY в переменных окружения Railway или в .env локально.")
        else:
            st.info("DeepSeek выключен. Включи в сайдбаре и добавь ключ в переменные окружения.")

    with main_tab6:
        st.subheader("📤 Export for TradingView")

        bundle = st.session_state.get("model_bundle")
        if bundle is None:
            st.info("Сначала обучи модели в табе ML, чтобы экспорт был не демо.")
            probs = {"large_move_probability": 0.50, "up_move_probability": 0.50, "down_move_probability": 0.50}
        else:
            probs = predict_latest(bundle, df_features)

        payload = build_signal_payload(
            df_features.iloc[-1],
            large_move_probability=float(probs["large_move_probability"]),
            up_move_probability=float(probs["up_move_probability"]),
            down_move_probability=float(probs["down_move_probability"]),
            symbol=symbol,
            timeframe=interval,
            notes="Market Intelligence Dashboard Export",
        )
        
        col_x, col_y = st.columns(2)
        with col_x:
            st.json(payload.__dict__)
        with col_y:
            bridge_text = save_tradingview_bridge(payload, Path("outputs/tradingview_ai_bridge.txt"))
            with open(bridge_text, "r", encoding="utf-8") as f:
                bridge_content = f.read()
            st.text_area("TradingView Bridge Payload", bridge_content, height=150)
            if st.button("Copy to Clipboard"):
                st.success("Payload ready to copy!")
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please check your internet connection or try a different symbol/timeframe.")
