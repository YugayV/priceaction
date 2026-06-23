from __future__ import annotations

import json
import sys
from pathlib import Path

from model_runtime import save_model_bundle, train_model_bundle
from run_pipeline import build_feature_frame, load_config, resolve_path


def main(config_path: str = "product_config.example.json") -> dict:
    base_dir = Path(__file__).resolve().parent
    config = load_config(resolve_path(base_dir, config_path))
    feature_df = build_feature_frame(config, base_dir)

    bundle, metrics = train_model_bundle(feature_df)
    models_dir = resolve_path(base_dir, config["models_dir"])
    save_paths = save_model_bundle(bundle, models_dir)

    metrics_path = models_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    result = {
        "status": "trained",
        "rows": int(len(feature_df)),
        "models_dir": str(models_dir),
        "saved_files": save_paths,
        "metrics_path": str(metrics_path),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    arg_path = sys.argv[1] if len(sys.argv) > 1 else "product_config.example.json"
    main(arg_path)
