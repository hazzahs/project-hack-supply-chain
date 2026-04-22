from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    data_dir: Path


def get_settings() -> AppSettings:
    project_root = Path(__file__).resolve().parent.parent
    return AppSettings(
        project_root=project_root,
        data_dir=project_root / "data",
    )
