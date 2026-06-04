from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_history(history: dict, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for key, values in history.items():
        if "loss" in key or "accuracy" in key:
            plt.plot(values, label=key)
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()


def plot_confusion_matrix(cm, class_names, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(cm)
    plt.figure(figsize=(7, 6))
    plt.imshow(arr)
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(arr[i, j]), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()


def plot_session(csv_path: str | Path, path: str | Path) -> None:
    df = pd.read_csv(csv_path)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(df["time_sec"], df["risk_score"], label="risk_score")
    plt.xlabel("Time, sec"); plt.ylabel("Risk"); plt.ylim(0, 1.05); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()
