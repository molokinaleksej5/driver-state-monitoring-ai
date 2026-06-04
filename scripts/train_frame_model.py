from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from driver_state.config import load_config
from driver_state.training.train_frame import train_frame_model
from driver_state.utils.seed import set_global_seed


if __name__ == "__main__":
    cfg = load_config()
    set_global_seed(cfg.seed)
    path = train_frame_model(cfg)
    print(f"Лучшая кадровая модель сохранена: {path}")
