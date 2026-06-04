from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


def project_root() -> Path:
    """Возвращает корень проекта независимо от того, откуда запущен скрипт."""
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "config" / "config.yaml"


@dataclass(frozen=True)
class AppConfig:
    seed: int
    image_size: Tuple[int, int]
    batch_size: int
    sequence_length: int
    sequence_stride: int
    class_names: List[str]
    dangerous_states: List[str]
    paths: Dict[str, str]
    mapping: Dict[str, str]
    training: Dict[str, float]
    decision: Dict[str, float]
    video: Dict[str, int]
    root_dir: Path

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def path(self, key: str) -> Path:
        value = Path(self.paths[key])
        if value.is_absolute():
            return value
        return self.root_dir / value


def _resolve_config_path(path: str | Path | None = None) -> Path:
    if path is None:
        return default_config_path()
    p = Path(path)
    if p.exists():
        return p.resolve()
    candidate = project_root() / p
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Не найден конфигурационный файл: {path}. "
        f"Ожидаемый путь: {default_config_path()}"
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    path = _resolve_config_path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    project = raw["project"]
    root_dir = path.parents[1]
    return AppConfig(
        seed=int(project["seed"]),
        image_size=tuple(project["image_size"]),
        batch_size=int(project["batch_size"]),
        sequence_length=int(project["sequence_length"]),
        sequence_stride=int(project["sequence_stride"]),
        class_names=list(project["class_names"]),
        dangerous_states=list(project["dangerous_states"]),
        paths=dict(raw["paths"]),
        mapping=dict(raw["state_farm_mapping"]),
        training=dict(raw["training"]),
        decision=dict(raw["decision"]),
        video=dict(raw["video"]),
        root_dir=root_dir,
    )


def ensure_project_dirs(cfg: AppConfig) -> None:
    for key in ["data_dir", "artifacts_dir", "models_dir", "reports_dir", "figures_dir", "logs_dir"]:
        cfg.path(key).mkdir(parents=True, exist_ok=True)
