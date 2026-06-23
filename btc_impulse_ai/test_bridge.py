
from pathlib import Path
from integration_bridge import SignalPayload, build_pine_bridge_payload, save_indicator_snapshot, save_tradingview_bridge
from datetime import datetime

# Создадим тестовый payload
test_payload = SignalPayload(
    timestamp=str(datetime.now()),
    symbol="BTC-USD",
    timeframe="15m",
    close=68500.0,
    large_move_probability=0.724,
    up_move_probability=0.648,
    down_move_probability=0.352,
    snr_bias="above_basis",
    session_state="new_york",
    kill_zone_state="london_kz",
    smc_state="bull_choch",
    entry_state="snr_retest_long",
    pa_signal="Bull Breakout",
    pa_bias="Long",
    pa_reason="Price broke structure or SNR resistance",
    pa_score=5.0,
    notes="Test payload"
)

# Сохраним файлы
outputs_dir = Path(__file__).parent / "outputs"
save_indicator_snapshot(test_payload, outputs_dir / "indicator_snapshot.json")
save_tradingview_bridge(test_payload, outputs_dir / "tradingview_ai_bridge.txt")

print("Test files saved!")
print("Bridge payload:", build_pine_bridge_payload(test_payload))
