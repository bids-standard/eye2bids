from __future__ import annotations

from pathlib import Path

import pytest


def data_dir() -> Path:
    return Path(__file__).parent / "data"


def asc_test_files(input_dir: Path = data_dir()) -> list[Path]:
    files = input_dir.glob("**/*.asc")
    tmp = [
        f
        for f in files
        if (not str(f).endswith("events.asc") and not str(f).endswith("samples.asc"))
    ]
    return tmp


def edf_test_files(input_dir: Path = data_dir()) -> list[Path]:
    files = list(input_dir.glob("**/*.edf"))
    EDF_files = list(input_dir.glob("**/*.EDF"))
    files.extend(EDF_files)
    return files


@pytest.fixture
def eyelink_test_data_dir() -> Path:
    return data_dir() / "osf" / "eyelink"
