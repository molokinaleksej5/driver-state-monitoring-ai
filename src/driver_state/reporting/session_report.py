from __future__ import annotations

from pathlib import Path
import pandas as pd

from driver_state.config import AppConfig
from driver_state.utils.io import save_json
from driver_state.visualization.plots import plot_session


def build_session_report(csv_path: str | Path, cfg: AppConfig, video_path: str | Path | None = None) -> dict:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        report = {"total_frames": 0, "csv_path": str(csv_path), "video_path": str(video_path) if video_path else None}
    else:
        dangerous = df[df["state"].isin(cfg.dangerous_states)]
        report = {
            "total_frames": int(len(df)),
            "face_detected_percent": round(float(df["face_detected"].mean() * 100), 2),
            "dangerous_frames": int(len(dangerous)),
            "dangerous_percent": round(float(len(dangerous) / len(df) * 100), 2),
            "warnings_count": int(df["warning"].sum()),
            "max_risk_score": round(float(df["risk_score"].max()), 4),
            "mean_risk_score": round(float(df["risk_score"].mean()), 4),
            "state_distribution_percent": df["state"].value_counts(normalize=True).mul(100).round(2).to_dict(),
            "csv_path": str(csv_path),
            "video_path": str(video_path) if video_path else None,
        }
    report_path = cfg.path("reports_dir") / (csv_path.stem + "_report.json")
    fig_path = cfg.path("figures_dir") / (csv_path.stem + "_risk.png")
    save_json(report, report_path)
    plot_session(csv_path, fig_path)
    report["report_path"] = str(report_path)
    report["figure_path"] = str(fig_path)
    return report
