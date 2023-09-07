import json
from pathlib import Path

import pytest

from eye2bids.edf2bids import (
    _check_edf2asc_present,
    _convert_edf_to_asc,
    _extract_CalibrationType,
    _load_asc_file_as_reduced_df,
    edf2bids,
)


def data_dir() -> Path:
    return Path(__file__).parent / "data"


def asc_test_files(input_dir: Path = data_dir()) -> list[Path]:
    return list(input_dir.glob("**/*.asc"))


def edf_test_files(input_dir: Path = data_dir()) -> list[Path]:
    files = list(input_dir.glob("**/*.edf"))
    EDF_files = list(input_dir.glob("**/*.EDF"))
    files.extend(EDF_files)
    return files


@pytest.mark.parametrize("input_file", edf_test_files())
def test_convert_edf_to_asc(input_file):
    if not _check_edf2asc_present():
        pytest.skip("edf2asc missing")
    asc_file = _convert_edf_to_asc(input_file)
    assert Path(asc_file).exists()


@pytest.mark.parametrize("metadata_file", [data_dir() / "metadata.yml", None])
def test_edf_end_to_end(metadata_file):
    if not _check_edf2asc_present():
        pytest.skip("edf2asc missing")

    input_dir = data_dir() / "osf" / "eyelink" / "decisions"
    input_file = edf_test_files(input_dir=input_dir)[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        metadata_file=metadata_file,
        output_dir=output_dir,
    )

    assert (output_dir / "events.json").exists()
    with open(output_dir / "events.json") as f:
        events = json.load(f)
    assert events["StimulusPresentation"]["ScreenResolution"] == [1919, 1079]

    assert (output_dir / "eyetrack.json").exists()
    with open(output_dir / "eyetrack.json") as f:
        eyetrack = json.load(f)
    assert eyetrack["SamplingFrequency"] == 1000
    assert eyetrack["RecordedEye"] == "Right"


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", "HV9"),
        ("emg", "HV9"),
        ("lt", "HV9"),
        ("pitracker", "HV9"),
        ("rest", "HV13"),
        ("satf", "HV9"),
        ("vergence", "HV9"),
    ],
)
def test_extract_CalibrationType(folder, expected):
    input_dir = data_dir() / "osf" / "eyelink" / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_CalibrationType(df_ms_reduced) == expected
