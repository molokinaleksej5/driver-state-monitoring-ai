from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from driver_state.config import load_config, ensure_project_dirs
from driver_state.inference.video_analyzer import VideoAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--attention", action="store_true", help="Включить Grad-CAM и прямоугольники зон внимания")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    source = args.camera if args.camera is not None else int(cfg.video["camera_id"])
    model_path = Path(args.model).resolve() if args.model else cfg.path("models_dir") / "frame_model_best.keras"
    analyzer = VideoAnalyzer(model_path, cfg, show_attention=args.attention)
    report = analyzer.analyze(source, save_video=True)
    print(report)


if __name__ == "__main__":
    main()
