from __future__ import annotations

from pathlib import Path
from warnings import warn

import pytest


def data_dir() -> Path:
    return Path(__file__).parent / "data"


def asc_test_files(input_dir: Path | None = None, suffix: str = "*") -> list[Path]:
    if input_dir is None:
        input_dir = data_dir()
    files = input_dir.glob(f"**/{suffix}.asc")
    tmp = list(files)
    if not tmp:
        warn(f"No .asc file found in: {input_dir}.", stacklevel=2)
    return tmp


def edf_test_files(input_dir: Path | None = None) -> list[Path]:
    if input_dir is None:
        input_dir = data_dir()
    files = list(input_dir.glob("**/*.edf"))
    edf_files = list(input_dir.glob("**/*.EDF"))
    files.extend(edf_files)
    if not files:
        warn(f"No EDF file found in: {input_dir}", stacklevel=2)
    return files


@pytest.fixture
def eyelink_test_data_dir() -> Path:
    return data_dir() / "osf" / "eyelink"
