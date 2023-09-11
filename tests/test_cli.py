from __future__ import annotations

from pathlib import Path

import pytest

from eye2bids._cli import cli
from eye2bids.edf2bids import _check_edf2asc_present

from .conftest import data_dir, edf_test_files


def root_dir() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).parent.parent


@pytest.mark.parametrize("metadata_file", [data_dir() / "metadata.yml", None])
@pytest.mark.parametrize("output_dir", [data_dir() / "output", None])
@pytest.mark.parametrize("use_relative_path", [False, True])
def test_edf_cli(use_relative_path, metadata_file, output_dir, eyelink_test_data_dir):
    if not _check_edf2asc_present():
        pytest.skip("edf2asc missing")

    input_dir = eyelink_test_data_dir / "decisions"
    input_file = edf_test_files(input_dir=input_dir)[0]

    if use_relative_path:
        input_file = input_file.relative_to(root_dir())

    command = ["eye2bids", "--input_file", str(input_file)]
    if metadata_file is not None:
        if use_relative_path:
            metadata_file = metadata_file.relative_to(root_dir())
        metadata_file = str(metadata_file)
        command.extend(["--metadata_file", metadata_file])

    if output_dir is not None:
        if use_relative_path:
            output_dir = output_dir.relative_to(root_dir())
        output_dir = str(output_dir)
        command.extend(["--output_dir", output_dir])

    cli(command)
