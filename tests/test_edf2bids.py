from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
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
    _df_events_after_start,
    _df_physioevents,
    _physioevents_eye1,
    _physioevents_eye2,
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

    expected_events_sidecar = (
        output_dir / f"{input_file.stem}_recording-eye1_physioevents.json"
    )
    assert expected_events_sidecar.exists()
    with open(expected_events_sidecar) as f:
        events = json.load(f)
    assert events["StimulusPresentation"]["ScreenResolution"] == [1919, 1079]

    expected_data_sidecar = output_dir / f"{input_file.stem}_recording-eye1_physio.json"
    assert expected_data_sidecar.exists()
    with open(expected_data_sidecar) as f:
        eyetrack = json.load(f)
    assert eyetrack["SamplingFrequency"] == 500
    assert eyetrack["RecordedEye"] == "Right"

    expected_data_tsv = output_dir / f"{input_file.stem}_recording-eye1_physio.tsv.gz"
    assert expected_data_tsv.exists()

    expected_events_tsv = output_dir / f"{input_file.stem}_recording-eye1_physioevents.tsv.gz"
    assert expected_events_tsv.exists()

@pytest.mark.skipif(not _check_edf2asc_present(), reason="edf2asc missing")
@pytest.mark.parametrize("metadata_file", [data_dir() / "metadata.yml", None])
def test_edf_end_to_end_2eyes(metadata_file, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / "2eyes"
    input_file = edf_test_files(input_dir=input_dir)[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        metadata_file=metadata_file,
        output_dir=output_dir,
    )

    expected_events_sidecar_eye1 = (
        output_dir / f"{input_file.stem}_recording-eye1_physioevents.json"
    )
    assert expected_events_sidecar_eye1.exists()
    with open(expected_events_sidecar_eye1) as f:
        events = json.load(f)
    assert events["StimulusPresentation"]["ScreenResolution"] == [1919, 1079]

    expected_data_sidecar_eye1 = output_dir / f"{input_file.stem}_recording-eye1_physio.json"
    assert expected_data_sidecar_eye1.exists()
    with open(expected_data_sidecar_eye1) as f:
        eyetrack = json.load(f)
    assert eyetrack["SamplingFrequency"] == 1000
    assert eyetrack["RecordedEye"] == "Left"

    expected_data_tsv_eye1 = output_dir / f"{input_file.stem}_recording-eye1_physio.tsv.gz"
    assert expected_data_tsv_eye1.exists()

    expected_events_tsv_eye1 = output_dir / f"{input_file.stem}_recording-eye1_physioevents.tsv.gz"
    assert expected_events_tsv_eye1.exists()

    expected_events_sidecar_eye2 = (
        output_dir / f"{input_file.stem}_recording-eye2_physioevents.json"
    )
    assert expected_events_sidecar_eye2.exists()

    expected_data_sidecar_eye2 = output_dir / f"{input_file.stem}_recording-eye2_physio.json"
    assert expected_data_sidecar_eye2.exists()
    with open(expected_data_sidecar_eye2) as f:
        eyetrack = json.load(f)
    assert eyetrack["RecordedEye"] == "Right"

    expected_data_tsv_eye2 = output_dir / f"{input_file.stem}_recording-eye2_physio.tsv.gz"
    assert expected_data_tsv_eye2.exists()

    expected_events_tsv_eye2 = output_dir / f"{input_file.stem}_recording-eye2_physioevents.tsv.gz"
    assert expected_events_tsv_eye2.exists()


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

    expected_eyetrack_tsv = output_dir / f"{input_file.stem}_recording-eye1_physio.tsv.gz"
    df = pd.read_csv(expected_eyetrack_tsv, sep="\t")
    count = sum(i == "." for i in df[0])
    assert count == 0


@pytest.mark.skipif(not _check_edf2asc_present(), reason="edf2asc missing")
def test_2files_eye1(eyelink_test_data_dir):
    """Check that for datafile with 2eyes 2 eye1 file is created and check input.

    function _2eyesmode
    """
    input_dir = eyelink_test_data_dir / "2eyes"
    input_file = edf_test_files(input_dir=input_dir)[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        output_dir=output_dir,
    )

    expected_eyetrack_sidecar = (
        output_dir / f"{input_file.stem}_recording-eye1_physio.json"
    )
    assert expected_eyetrack_sidecar.exists()
    with open(expected_eyetrack_sidecar) as f:
        eyetrack = json.load(f)
    assert eyetrack["AverageCalibrationError"] == [[0.29]]
    assert eyetrack["RecordedEye"] == "Left"


@pytest.mark.skipif(not _check_edf2asc_present(), reason="edf2asc missing")
def test_2files_eye2(eyelink_test_data_dir):
    """Check that for datafile with 2eyes 2 eye2 file is created and check input.

    function _2eyesmode
    """
    input_dir = eyelink_test_data_dir / "2eyes"
    input_file = edf_test_files(input_dir=input_dir)[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        output_dir=output_dir,
    )

    expected_eyetrack_sidecar = (
        output_dir / f"{input_file.stem}_recording-eye2_physio.json"
    )
    assert expected_eyetrack_sidecar.exists()
    with open(expected_eyetrack_sidecar) as f:
        eyetrack = json.load(f)
    assert eyetrack["AverageCalibrationError"] == [[0.35]]
    assert eyetrack["RecordedEye"] == "Right"


@pytest.mark.skipif(not _check_edf2asc_present(), reason="edf2asc missing")
def test_2files_eye1(eyelink_test_data_dir):
    """Check that for datafile with 2eyes 2 eye1 file is created and check input.

    function _2eyesmode
    """
    input_dir = eyelink_test_data_dir / "2eyes"
    input_file = edf_test_files(input_dir=input_dir)[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        output_dir=output_dir,
    )

    expected_eyetrack_sidecar = (
        output_dir / f"{input_file.stem}_recording-eye1_physio.json"
    )
    assert expected_eyetrack_sidecar.exists()
    with open(expected_eyetrack_sidecar) as f:
        eyetrack = json.load(f)
    assert eyetrack["AverageCalibrationError"] == [[0.29]]
    assert eyetrack["RecordedEye"] == "Left"


@pytest.mark.skipif(not _check_edf2asc_present(), reason="edf2asc missing")
def test_2files_eye2(eyelink_test_data_dir):
    """Check that for datafile with 2eyes 2 eye2 file is created and check input.

    function _2eyesmode
    """
    input_dir = eyelink_test_data_dir / "2eyes"
    input_file = edf_test_files(input_dir=input_dir)[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        output_dir=output_dir,
    )

    expected_eyetrack_sidecar = (
        output_dir / f"{input_file.stem}_recording-eye2_physio.json"
    )
    assert expected_eyetrack_sidecar.exists()
    with open(expected_eyetrack_sidecar) as f:
        eyetrack = json.load(f)
    assert eyetrack["AverageCalibrationError"] == [[0.35]]
    assert eyetrack["RecordedEye"] == "Right"


@pytest.mark.skipif(not _check_edf2asc_present(), reason="edf2asc missing")
def test_number_columns_2eyes_tsv(eyelink_test_data_dir):
    """Check that values for only one eye were extracted in eye1-physio.tsv.gz by number of columns.

    function _samples_to_data_frame
    """
    input_dir = eyelink_test_data_dir / "2eyes"
    input_file = edf_test_files(input_dir=input_dir)[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        output_dir=output_dir,
    )

    expected_eyetrack_tsv = output_dir / f"{input_file.stem}_recording-eye1_physio.tsv.gz"
    df = pd.read_csv(expected_eyetrack_tsv, sep="\t")
    number_columns = len(df.columns)
    assert number_columns == 4


@pytest.mark.skipif(not _check_edf2asc_present(), reason="edf2asc missing")
def test_number_columns_1eye_tsv(eyelink_test_data_dir):
    """Check that values for one eye were extracted by number of columns.

    function _samples_to_data_frame.
    """
    input_dir = eyelink_test_data_dir / "rest"
    print(edf_test_files(input_dir=input_dir))
    input_file = edf_test_files(input_dir=input_dir)[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        output_dir=output_dir,
    )

    expected_eyetrack_tsv = output_dir / f"{input_file.stem}_recording-eye1_physio.tsv.gz"
    df = pd.read_csv(expected_eyetrack_tsv, sep="\t")
    number_columns = len(df.columns)
    assert number_columns == 4


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("emg", "HV9"),
        ("lt", "HV9"),
        ("pitracker", "HV9"),
        ("rest", "HV13"),
        ("satf", "HV9"),
        ("vergence", "HV9"),
        ("2eyes", "HV13"),
    ],
)
def test_extract_CalibrationType(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_CalibrationType(df_ms_reduced) == expected


# ("rest", "FIXME"),
@pytest.mark.parametrize(
    "folder, expected",
    [
        ("emg", [1919, 1079]),
        ("lt", [1919, 1079]),
        ("pitracker", [1919, 1079]),
        ("satf", [1919, 1079]),
        ("vergence", [1919, 1079]),
        ("2eyes", [1919, 1079]),
    ],
)
def test_extract_ScreenResolution(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_ScreenResolution(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("emg", ""),
        ("lt", "pixel"),
        ("pitracker", ""),
        ("rest", "pixel"),
        ("satf", ""),
        ("vergence", ""),
        ("2eyes", "pixel"),
    ],
)
def test_extract_CalibrationUnit(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_CalibrationUnit(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
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
                    [576, 540],
                    [1344, 540],
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
        (
            "2eyes",
            [
                [
                    [960, 540],
                    [960, 732],
                    [1126, 444],
                    [576, 540],
                    [1344, 540],
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
    ],
)
def test_extract_CalibrationPosition(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_CalibrationPosition(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("emg", "P-CR"),
        ("lt", "P-CR"),
        ("pitracker", "P-CR"),
        ("rest", "P-CR"),
        ("satf", "P-CR"),
        ("vergence", "P-CR"),
        ("2eyes", "P-CR"),
    ],
)
def test_extract_EyeTrackingMethod(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    events = _load_asc_file(asc_file)
    assert _extract_EyeTrackingMethod(events) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("emg", 1000),
        ("lt", 1000),
        ("pitracker", 1000),
        ("rest", 1000),
        ("satf", 500),
        ("vergence", 1000),
        ("2eyes", 1000),
    ],
)
def test_extract_SamplingFrequency(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_SamplingFrequency(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("emg", "ELLIPSE"),
        ("lt", "CENTROID"),
        ("pitracker", "CENTROID"),
        ("rest", "CENTROID"),
        ("satf", "CENTROID"),
        ("vergence", "CENTROID"),
        ("2eyes", "CENTROID"),
    ],
)
def test_extract_PupilFitMethod(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_PupilFitMethod(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("emg", "CLG-BBF01"),
        ("lt", "CLG-BCC29"),
        ("pitracker", "CLG-BAF22"),
        ("rest", "CLO-ZBD04"),
        ("satf", "CL1-ACF05"),
        ("vergence", "CL1-72N02"),
        ("2eyes", "CLG-BAF38"),
    ],
)
def test_extract_DeviceSerialNumber(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    events = _load_asc_file(asc_file)
    assert _extract_DeviceSerialNumber(events) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("emg", "Right"),
        ("lt", "Left"),
        ("pitracker", "Right"),
        ("rest", "Left"),
        ("satf", "Right"),
        ("vergence", ["Left", "Right"]),
        ("2eyes", ["Left", "Right"]),
        ("vergence", ["Left", "Right"]),
        ("2eyes", ["Left", "Right"]),
    ],
)
def test_extract_RecordedEye(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    df_ms_reduced = _load_asc_file_as_reduced_df(asc_file)
    assert _extract_RecordedEye(df_ms_reduced) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("emg", "EYELINK II CL v5.04 Sep 25 2014"),
        ("lt", "EYELINK II CL v5.15 Jan 24 2018"),
        ("pitracker", "EYELINK II CL v5.01 Jan 16 2014"),
        ("rest", "EYELINK II CL v5.09 Nov 17 2015"),
        ("satf", "EYELINK II CL v4.594 Jul  6 2012"),
        ("vergence", "EYELINK II CL v4.56 Aug 18 2010"),
        ("2eyes", "EYELINK II CL v5.12 May 12 2017"),
    ],
)
def test_extract_ManufacturersModelName(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    events = _load_asc_file(asc_file)
    assert _extract_ManufacturersModelName(events) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("emg", []),
        ("lt", [[0.32], [0.37]]),
        ("pitracker", []),
        ("rest", [[0.9]]),
        ("satf", []),
        ("vergence", []),
        (
            "2eyes",
            [[0.62], [1.21]],
        ),
    ],
)
def test_extract_MaximalCalibrationError(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    df_ms = _load_asc_file_as_df(asc_file)
    assert _extract_MaximalCalibrationError(df_ms) == expected


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("emg", []),
        ("lt", [[0.16], [0.18]]),
        ("pitracker", []),
        ("rest", [[0.65]]),
        ("satf", []),
        ("vergence", []),
        (
            "2eyes",
            [[0.29], [0.35]],
        ),
    ],
)
def test_extract_AverageCalibrationError(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    df_ms = _load_asc_file_as_df(asc_file)
    assert _extract_AverageCalibrationError(df_ms) == expected


@pytest.mark.skipif(not _check_edf2asc_present(), reason="edf2asc missing")
def test_number_columns_physioevents_tsv(eyelink_test_data_dir):
    """Check right number of columns in physioevents.tsv.gz.

    function _df_physioevents, _physioevents_eye1, _physioevents_eye2
    """
    input_dir = eyelink_test_data_dir / "2eyes"
    print(edf_test_files(input_dir=input_dir))
    input_file = edf_test_files(input_dir=input_dir)[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        output_dir=output_dir,
    )

    expected_physioevents_tsv = output_dir / f"{input_file.stem}_recording-eye2_physioevents.tsv.gz"
    df = pd.read_csv(expected_physioevents_tsv, sep="\t")
    number_columns = len(df.columns)
    assert number_columns == 5

@pytest.mark.parametrize(
    "folder, expected",
    [
        ("rest", ['fixation', 'saccade', 'fixation', 'saccade', 'fixation', 'saccade', 'fixation']),
        ("2eyes", ['fixation', 'saccade', 'fixation', 'saccade', 'fixation', 'saccade', 'fixation']),
        ("pitracker", ['saccade', 'fixation', 'saccade', 'fixation', 'saccade', 'fixation', 'saccade']),
    ],
)
def test_physioevents_value(folder, expected, eyelink_test_data_dir):
    input_dir = eyelink_test_data_dir / folder
    asc_file = asc_test_files(input_dir=input_dir, suffix="*_events")[0]
    events = _load_asc_file(asc_file)
    events_after_start = _df_events_after_start(events)
    physioevents_reordered = _df_physioevents(events_after_start)
    physioevents_eye1 = _physioevents_eye1(physioevents_reordered)
    assert physioevents_eye1.iloc[3:10, 2].tolist() == expected