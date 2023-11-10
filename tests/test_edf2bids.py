from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from eye2bids.edf2bids import (
    _check_edf2asc_present,
    _convert_edf_to_asc_events,
    _extract_AverageCalibrationError,
    _extract_CalibrationPosition,
    _extract_CalibrationType,
    _extract_CalibrationUnit,
    _extract_DeviceSerialNumber,
    _extract_EyeTrackingMethod,
    _extract_ManufacturersModelName,
    _extract_MaximalCalibrationError,
    _extract_PupilFitMethod,
    _extract_RecordedEye,
    _extract_SamplingFrequency,
    _extract_ScreenResolution,
    _load_asc_file,
    _load_asc_file_as_df,
    _load_asc_file_as_reduced_df,
    edf2bids,
)

from .conftest import asc_test_files, data_dir, edf_test_files


@pytest.mark.skipif(not _check_edf2asc_present(), reason="edf2asc missing")
@pytest.mark.parametrize("input_file", edf_test_files())
def test_convert_edf_to_asc_events(input_file):
    asc_file = _convert_edf_to_asc_events(input_file)
    assert Path(asc_file).exists()


@pytest.mark.skipif(not _check_edf2asc_present(), reason="edf2asc missing")
@pytest.mark.parametrize("metadata_file", [data_dir() / "metadata.yml", None])
def test_edf_end_to_end(metadata_file, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / "satf"
    input_file = edf_test_files(input_dir=input_dir)[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        metadata_file=metadata_file,
        output_dir=output_dir,
    )

    expected_events_sidecar = output_dir / f"{input_file.stem}_events.json"
    assert expected_events_sidecar.exists()
    with open(expected_events_sidecar) as f:
        events = json.load(f)
    assert events["StimulusPresentation"]["ScreenResolution"] == [1919, 1079]

    expected_data_sidecar = output_dir / f"{input_file.stem}_eyetrack.json"
    assert expected_data_sidecar.exists()
    with open(expected_data_sidecar) as f:
        eyetrack = json.load(f)
    assert eyetrack["SamplingFrequency"] == 500
    assert eyetrack["RecordedEye"] == "Right"

    expected_events_sidecar = output_dir / f"{input_file.stem}_eyetrack.tsv"
    assert expected_events_sidecar.exists()


@pytest.mark.skipif(not _check_edf2asc_present(), reason="edf2asc missing")
def test_edf_nan_in_tsv(eyelink_test_data_dir):
    """Check that dots '.' are converted to NaN in the tsv file."""
    input_dir = eyelink_test_data_dir / "emg"
    input_file = edf_test_files(input_dir=input_dir)[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        output_dir=output_dir,
    )

    expected_events_sidecar = output_dir / f"{input_file.stem}_eyetrack.tsv"
    df = pd.read_csv(expected_events_sidecar, sep="\t")
    count = sum(i == "." for i in df["eye1_x_coordinate"])
    assert count == 0


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
def test_extract_CalibrationType(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_CalibrationType(df_ms_reduced) == expected


# ("rest", "FIXME"),
@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", [1919, 1079]),
        ("emg", [1919, 1079]),
        ("lt", [1919, 1079]),
        ("pitracker", [1919, 1079]),
        ("satf", [1919, 1079]),
        ("vergence", [1919, 1079]),
    ],
)
def test_extract_ScreenResolution(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
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
def test_extract_CalibrationUnit(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
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
def test_extract_CalibrationPosition(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
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
def test_extract_EyeTrackingMethod(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
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
def test_extract_SamplingFrequency(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_SamplingFrequency(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", "ELLIPSE"),
        ("emg", "ELLIPSE"),
        ("lt", "CENTROID"),
        ("pitracker", "CENTROID"),
        ("rest", "CENTROID"),
        ("satf", "CENTROID"),
        ("vergence", "CENTROID"),
    ],
)
def test_extract_PupilFitMethod(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_PupilFitMethod(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", "CLG-BBF01"),
        ("emg", "CLG-BBF01"),
        ("lt", "CLG-BCC29"),
        ("pitracker", "CLG-BAF22"),
        ("rest", "CLO-ZBD04"),
        ("satf", "CL1-ACF05"),
        ("vergence", "CL1-72N02"),
    ],
)
def test_extract_DeviceSerialNumber(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    events = _load_asc_file(asc_file)
    assert _extract_DeviceSerialNumber(events) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", "Right"),
        ("emg", "Right"),
        ("lt", "Left"),
        ("pitracker", "Right"),
        ("rest", "Left"),
        ("satf", "Right"),
        ("vergence", "Both"),
    ],
)
def test_extract_RecordedEye(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_RecordedEye(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", "EYELINK II CL v5.04 Sep 25 2014"),
        ("emg", "EYELINK II CL v5.04 Sep 25 2014"),
        ("lt", "EYELINK II CL v5.15 Jan 24 2018"),
        ("pitracker", "EYELINK II CL v5.01 Jan 16 2014"),
        ("rest", "EYELINK II CL v5.09 Nov 17 2015"),
        ("satf", "EYELINK II CL v4.594 Jul  6 2012"),
        ("vergence", "EYELINK II CL v4.56 Aug 18 2010"),
    ],
)
def test_extract_ManufacturersModelName(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    events = _load_asc_file(asc_file)
    assert _extract_ManufacturersModelName(events) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", []),
        ("emg", []),
        ("lt", [[0.32], [0.37]]),
        ("pitracker", []),
        ("rest", [[0.9]]),
        ("satf", []),
        ("vergence", []),
    ],
)
def test_extract_MaximalCalibrationError(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    df_ms = _load_asc_file_as_df(asc_file)
    assert _extract_MaximalCalibrationError(df_ms) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("decisions", []),
        ("emg", []),
        ("lt", [[0.16], [0.18]]),
        ("pitracker", []),
        ("rest", [[0.65]]),
        ("satf", []),
        ("vergence", []),
    ],
)
def test_extract_AverageCalibrationError(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir)[0]
    df_ms = _load_asc_file_as_df(asc_file)
    assert _extract_AverageCalibrationError(df_ms) == expected
