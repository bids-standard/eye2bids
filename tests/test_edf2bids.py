import json
from pathlib import Path

import pytest

from eye2bids.edf2bids import (
    _check_edf2asc_present,
    _convert_edf_to_asc,
    _extract_CalibrationPosition,
    _extract_CalibrationType,
    _extract_CalibrationUnit,
    _extract_EyeTrackingMethod,
    _extract_SamplingFrequency,
    _extract_ScreenResolution,
    _load_asc_file,
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


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", [1919, 1079]),
        ("emg", [1919, 1079]),
        ("lt", [1919, 1079]),
        ("pitracker", [1919, 1079]),
        ("rest", "FIXME"),
        ("satf", [1919, 1079]),
        ("vergence", [1919, 1079]),
    ],
)
def test_extract_ScreenResolution(folder, expected):
    input_dir = data_dir() / "osf" / "eyelink" / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_ScreenResolution(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", ""),
        ("emg", ""),
        ("lt", "pixel"),
        ("pitracker", ""),
        ("rest", "pixel"),
        ("satf", ""),
        ("vergence", ""),
    ],
)
def test_extract_CalibrationUnit(folder, expected):
    input_dir = data_dir() / "osf" / "eyelink" / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_CalibrationUnit(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", []),
        ("emg", []),
        (
            "lt",
            [
                [
                    [960, 540],
                    [960, 324],
                    [960, 755],
                    [576, 540],
                    [1343, 540],
                    [622, 350],
                    [1297, 350],
                    [622, 729],
                    [1297, 729],
                ],
                [
                    [960, 540],
                    [960, 324],
                    [960, 755],
                    [576, 540],
                    [1343, 540],
                    [622, 350],
                    [1297, 350],
                    [622, 729],
                    [1297, 729],
                ],
            ],
        ),
        ("pitracker", []),
        (
            "rest",
            [
                [
                    [960, 540],
                    [960, 732],
                    [1126, 444],
                    [1344, 540],
                    [576, 540],
                    [768, 873],
                    [1152, 873],
                    [768, 207],
                    [1152, 207],
                    [794, 636],
                    [1126, 636],
                    [794, 444],
                    [960, 348],
                ]
            ],
        ),
        ("satf", []),
        ("vergence", []),
    ],
)
def test_extract_CalibrationPosition(folder, expected):
    input_dir = data_dir() / "osf" / "eyelink" / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_CalibrationPosition(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", "P-CR"),
        ("emg", "P-CR"),
        ("lt", "P-CR"),
        ("pitracker", "P-CR"),
        ("rest", "P-CR"),
        ("satf", "P-CR"),
        ("vergence", "P-CR"),
    ],
)
def test_extract_EyeTrackingMethod(folder, expected):
    input_dir = data_dir() / "osf" / "eyelink" / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    events = _load_asc_file(asc_file)
    assert _extract_EyeTrackingMethod(events) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", 1000),
        ("emg", 1000),
        ("lt", 1000),
        ("pitracker", 1000),
        ("rest", 1000),
        ("satf", 500),
        ("vergence", 1000),
    ],
)
def test_extract_SamplingFrequency(folder, expected):
    input_dir = data_dir() / "osf" / "eyelink" / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_SamplingFrequency(df_ms_reduced) == expected