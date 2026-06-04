from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from driver_state.config import load_config, ensure_project_dirs
from driver_state.evaluation.metrics import evaluate_frame_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    model_path = Path(args.model).resolve() if args.model else cfg.path("models_dir") / "frame_model_best.keras"
    result = evaluate_frame_model(model_path, cfg, args.split)
    print(result)
