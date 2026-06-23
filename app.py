
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

from snr_ml_features import FeatureConfig, add_snr_features, load_market_data
from integration_bridge import build_signal_payload, save_indicator_snapshot, save_tradingview_bridge
from news_analyzer import NewsAnalyzer
from deepseek_analyzer import DeepSeekAnalyzer

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

try:
    data_load_state = st.text("Loading market data...")
    df = get_market_data(symbol, interval, period, timezone)
    data_load_state.text("")
    
    if refresh_data:
        st.cache_data.clear()
        df = get_market_data(symbol, interval, period, timezone)
    
    df_features = get_features(df)
    
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
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "📈 Price Action",
        "📰 News Analysis",
        "🤖 DeepSeek Analysis",
        "📤 Export"
    ])
    
    # Price Action Tab
    with main_tab1:
        # Price Chart with SNR
        st.subheader("Price Action with SNR")
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ))
        
        # SNR Lines
        fig.add_trace(go.Scatter(
            x=df_features.index,
            y=df_features["reg_basis"],
            mode="lines",
            name="SNR Basis",
            line=dict(color="#00ff88", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df_features.index,
            y=df_features["snr_upper"],
            mode="lines",
            name="SNR Upper",
            line=dict(color="#ff4444", width=1, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=df_features.index,
            y=df_features["snr_lower"],
            mode="lines",
            name="SNR Lower",
            line=dict(color="#44ff44", width=1, dash="dash"),
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=600,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Features and Metrics Subtabs
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🎯 Price Action Features", "📊 Market Structure", "📉 Indicators"])
        
        with sub_tab1:
            st.subheader("Price Action & Flow")
            col_a, col_b = st.columns(2)
            with col_a:
                # Flow Chart
                fig_flow = go.Figure()
                fig_flow.add_trace(go.Scatter(
                    x=df_features.index,
                    y=df_features["flow_main"],
                    mode="lines",
                    name="Flow Main",
                    line=dict(color="#8888ff"),
                ))
                fig_flow.add_hline(y=0, line_dash="dash", line_color="#ffffff")
                fig_flow.update_layout(
                    title="Flow Indicator",
                    template="plotly_dark",
                    height=400,
                )
                st.plotly_chart(fig_flow, use_container_width=True)
            with col_b:
                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=df_features.index,
                    y=df_features["rsi"],
                    mode="lines",
                    name="RSI",
                    line=dict(color="#ff8888"),
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff4444")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="#44ff44")
                fig_rsi.update_layout(
                    title="RSI (14)",
                    template="plotly_dark",
                    height=400,
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
        
        with sub_tab2:
            st.subheader("Market Structure & Sessions")
            # Session Activity
            session_counts = {
                "Asia": df_features["is_asia"].sum(),
                "London": df_features["is_london"].sum(),
                "New York": df_features["is_ny"].sum(),
            }
            fig_sessions = px.bar(
                x=list(session_counts.keys()),
                y=list(session_counts.values()),
                title="Session Activity",
                color=list(session_counts.keys()),
                color_discrete_map={
                    "Asia": "#3399ff",
                    "London": "#00a36c",
                    "New York": "#d2445a",
                },
                template="plotly_dark",
            )
            st.plotly_chart(fig_sessions, use_container_width=True)
        
        with sub_tab3:
            st.subheader("Technical Indicators")
            st.dataframe(df_features[["open", "high", "low", "close", "volume", "rsi", "atr14", "reg_basis", "kz_long_score", "kz_short_score"]].tail(20), use_container_width=True)
    
    # News Analysis Tab
    with main_tab2:
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
    
    # DeepSeek Analysis Tab
    with main_tab3:
        if enable_deepseek:
            st.subheader("🤖 DeepSeek AI Analysis")
            if st.button("Generate Analysis"):
                with st.spinner("Analyzing market with DeepSeek AI..."):
                    try:
                        # Prepare price data
                        price_data = {
                            "symbol": symbol,
                            "latest_close": float(latest_close),
                            "price_change": float(price_change),
                            "volume": float(latest_vol),
                            "rsi": float(df_features['rsi'].iloc[-1]),
                            "atr": float(df_features['atr14'].iloc[-1]),
                        }
                        
                        # Get news
                        news_data = get_news(asset_key) if enable_news else []
                        
                        # Analyze with DeepSeek
                        deepseek = DeepSeekAnalyzer()
                        analysis = deepseek.analyze_market(asset_key, price_data, news_data)
                        
                        st.markdown(analysis)
                    except Exception as e:
                        st.error(f"Error during DeepSeek analysis: {str(e)}")
                        st.info("Make sure you have set your DEEPSEEK_API_KEY in .env file")
        else:
            st.info("DeepSeek analysis is disabled. Enable it in the sidebar and set your API key.")
            st.markdown("""
            To use DeepSeek analysis:
            1. Create a `.env` file from `.env.example`
            2. Add your DeepSeek API key
            3. Enable the feature in the sidebar
            """)
    
    # Export Tab
    with main_tab4:
        st.subheader("📤 Export for TradingView")
        
        # Simulate some predictions for demo
        st.info("Note: AI predictions require a trained model. This is a demo export.")
        
        demo_payload = build_signal_payload(
            df_features.iloc[-1],
            large_move_probability=0.65,
            up_move_probability=0.72,
            down_move_probability=0.28,
            symbol=symbol,
            timeframe=interval,
            notes="Market Intelligence Dashboard Export",
        )
        
        col_x, col_y = st.columns(2)
        with col_x:
            st.json(demo_payload.__dict__)
        with col_y:
            bridge_text = save_tradingview_bridge(demo_payload, Path("outputs/tradingview_ai_bridge.txt"))
            with open(bridge_text, "r", encoding="utf-8") as f:
                bridge_content = f.read()
            st.text_area("TradingView Bridge Payload", bridge_content, height=150)
            if st.button("Copy to Clipboard (Demo)"):
                st.success("Payload ready to copy!")
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please check your internet connection or try a different symbol/timeframe.")

