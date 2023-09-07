"""Main module for conversion of edf to bids compliant files."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader

from eye2bids._parser import global_parser


def _check_inputs(
    input_file: str | Path | None = None,
    metadata_file: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    if input_file is None:
        input_file = input("Enter the edf file path: ")
    if isinstance(input_file, str):
        input_file = Path(input_file)
    if input_file.exists():
        print("The file exists")
    else:
        raise FileNotFoundError(f"No such file: {input_file}")

    if metadata_file is None:
        # Read variables from the additional metadata txt file
        print(
            """Load the metadata.yml file with the additional metadata.\n
    This file must contain at least the additional REQUIRED metadata
    in the format specified in the BIDS specification.\n
    Please enter the required metadata manually
    before loading the file in a next step."""
        )
        metadata_file = input("Enter the file path to the metadata.yml file: ")
    if isinstance(metadata_file, str):
        metadata_file = Path(metadata_file)
    if isinstance(metadata_file, str):
        metadata_file = Path(metadata_file)
    if metadata_file.exists():
        print("The file exists")
    else:
        raise FileNotFoundError(f"No such file: {metadata_file}")

    if output_dir is None:
        output_dir = input("Enter the output directory: ")
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    return input_file, metadata_file, output_dir


def _check_edf2asc_present() -> bool:
    """Check if edf2asc is present in the path."""
    try:
        subprocess.run(["edf2asc"])
        return True
    except FileNotFoundError:
        print("edf2asc not found in path")
        return False


def main(
    input_file: str | Path | None = None,
    metadata_file: str | Path | None = None,
    output_dir: str | Path | None = None,
):
    """Convert edf to tsv + json."""
    if not _check_edf2asc_present():
        return

    input_file, metadata_file, output_dir = _check_inputs(
        input_file, metadata_file, output_dir
    )

    # CONVERSION events
    subprocess.run(["edf2asc", "-y", "-e", input_file, f"{str(input_file)}_events"])

    # Prepare asc file
    asc_file = f"{str(input_file)}_events.asc"
    with open(asc_file) as f:
        events = f.readlines()

    # dataframe for events, all
    df_ms = pd.DataFrame([ms.split() for ms in events if ms.startswith("MSG")])

    # reduced dataframe without MSG and sample columns
    df_ms_reduced = pd.DataFrame(df_ms.iloc[0:, 2:])

    # Eyetrack.json Metadata
    # TODO:figure out if this is actually the StartTime meant by the specification
    StartTime = (
        np.array(pd.DataFrame([st.split() for st in events if st.startswith("START")])[1])
        .astype(int)
        .tolist()
    )

    # TODO:figure out if this is actually the StopTime meant by the specification
    StopTime = (
        np.array(pd.DataFrame([so.split() for so in events if so.startswith("END")])[1])
        .astype(int)
        .tolist()
    )

    with open(metadata_file) as f:
        metadata = yaml.load(f, Loader=SafeLoader)

    # to json
    eyetrack_json = {
        "Manufacturer": "SR-Research",
        "EnvironmentCoordinates": metadata["EnvironmentCoordinates"],
        "EyeCameraSettings": metadata["EyeCameraSettings"],
        "EyeTrackerDistance": metadata["EyeTrackerDistance"],
        "FeatureDetectionSettings": metadata["FeatureDetectionSettings"],
        "GazeMappingSettings": metadata["GazeMappingSettings"],
        "RawDataFilters": metadata["RawDataFilters"],
        "SampleCoordinateSystem": metadata["SampleCoordinateSystem"],
        "SampleCoordinateUnits": metadata["SampleCoordinateUnits"],
        "ScreenAOIDefinition": metadata["ScreenAOIDefinition"],
        "SoftwareVersion": metadata["SoftwareVersion"],
        "DeviceSerialNumber": _extract_DeviceSerialNumber(events),
        "EyeTrackingMethod": _extract_EyeTrackingMethod(events),
        "ManufacturersModelName": _extract_ManufacturersModelName(events),
        "AverageCalibrationError": _extract_AverageCalibrationError(df_ms),
        "MaximalCalibrationError": _extract_MaximalCalibrationError(df_ms),
        "CalibrationCount": _extract_CalibrationCount(df_ms_reduced),
        "CalibrationPosition": _extract_CalibrationPosition(df_ms_reduced),
        "CalibrationUnit": _extract_CalibrationUnit(df_ms_reduced),
        "CalibrationType": _extract_CalibrationType(df_ms_reduced),
        "PupilFitMethod": _extract_PupilFitMethod(df_ms_reduced),
        "RecordedEye": _extract_RecordedEye(df_ms_reduced),
        "SamplingFrequency": _extract_SamplingFrequency(df_ms_reduced),
        "StartTime": StartTime,
        "StopTime": StopTime,
    }

    with open(output_dir / "eyetrack.json", "w") as outfile:
        json.dump(eyetrack_json, outfile, indent=4)

    # Events.json Metadata
    events_json = {
        "InstitutionAddress": metadata["InstitutionAddress"],
        "InstitutionName": metadata["InstitutionName"],
        "StimulusPresentation": {
            "ScreenDistance": metadata["ScreenDistance"],
            "ScreenRefreshRate": metadata["ScreenRefreshRate"],
            "ScreenSize": metadata["ScreenSize"],
            "ScreenResolution": _extract_ScreenResolution(df_ms_reduced),
        },
        "TaskName": _extract_TaskName(events),
    }

    with open(output_dir / "events.json", "w") as outfile:
        json.dump(events_json, outfile, indent=4)


def _calibrations(df):
    return df[df[3] == "CALIBRATION"]


def _extract_CalibrationType(df: pd.DataFrame) -> list[int]:
    return _calibrations(df).iloc[0:1, 2:3].to_string(header=False, index=False)


def _extract_CalibrationCount(df: pd.DataFrame) -> int:
    return len(_calibrations(df))


def _get_calibration_positions(df: pd.DataFrame) -> np.array:
    return (
        np.array(df[df[2] == "VALIDATE"][8].str.split(",", expand=True))
        .astype(int)
        .tolist()
    )


def _extract_CalibrationPosition(df: pd.DataFrame) -> list[list[int]]:
    cal_pos = _get_calibration_positions(df)
    cal_num = len(cal_pos) // _extract_CalibrationCount(df)

    CalibrationPosition = []

    if len(cal_pos) == 0:
        return CalibrationPosition

    CalibrationPosition.extend(
        cal_pos[i : i + cal_num] for i in range(0, len(cal_pos), cal_num)
    )
    return CalibrationPosition


def _extract_CalibrationUnit(df: pd.DataFrame) -> str:
    if len(_get_calibration_positions(df)) == 0:
        return ""

    cal_unit = (
        (df[df[2] == "VALIDATE"][[13]])
        .iloc[0:1, 0:1]
        .to_string(header=False, index=False)
    )
    if cal_unit in ["cm", "mm"]:
        return "cm"
    elif cal_unit == "pix.":
        return "pixel"


def _extract_EyeTrackingMethod(events: list[str]) -> str:
    return (
        pd.DataFrame(
            " ".join([tm for tm in events if tm.startswith(">>>>>>>")])
            .replace(")", ",")
            .split(",")
        )
        .iloc[1:2]
        .to_string(header=False, index=False)
    )


def _validations(df: pd.DataFrame):
    return df[df[3] == "VALIDATION"]


def _has_validation(df: pd.DataFrame) -> bool:
    return not _validations(df).empty


def _extract_MaximalCalibrationError(df: pd.DataFrame) -> list[float]:
    if not _has_validation(df):
        return []
    return np.array(_validations(df)[[11]]).astype(float).tolist()


def _extract_AverageCalibrationError(df: pd.DataFrame) -> list[float]:
    if _has_validation(df):
        return []
    return np.array(_validations(df)[[9]]).astype(float).tolist()


def _extract_ManufacturersModelName(events: list[str]) -> str:
    return (
        " ".join([ml for ml in events if ml.startswith("** EYELINK")])
        .replace("** ", "")
        .replace("\n", "")
    )


def _extract_DeviceSerialNumber(events: list[str]) -> str:
    return (
        " ".join([sl for sl in events if sl.startswith("** SERIAL NUMBER:")])
        .replace("** SERIAL NUMBER: ", "")
        .replace("\n", "")
    )


def _extract_PupilFitMethod(df: pd.DataFrame) -> str:
    return (df[df[2] == "ELCL_PROC"]).iloc[0:1, 1:2].to_string(header=False, index=False)


def _extract_SamplingFrequency(df: pd.DataFrame) -> int:
    return int(df[df[2] == "RECCFG"].iloc[0:1, 2:3].to_string(header=False, index=False))


def _extract_RecordedEye(df: pd.DataFrame) -> str:
    eye = df[df[2] == "RECCFG"].iloc[0:1, 5:6].to_string(header=False, index=False)
    if eye == "L":
        return "Left"
    elif eye == "R":
        return "Right"
    elif eye == "LR":
        return "Both"


def _extract_ScreenResolution(df: pd.DataFrame) -> list[int]:
    return list(
        map(
            int,
            (
                df[df[2] == "DISPLAY_COORDS"]
                .iloc[0:1, 3:5]
                .to_string(header=False, index=False)
            ).split(" "),
        )
    )


def _extract_TaskName(events: list[str]):
    return (
        " ".join([ts for ts in events if ts.startswith("** RECORDED BY")])
        .replace("** RECORDED BY ", "")
        .replace("\n", "")
    )


if __name__ == "__main__":
    parser = global_parser()
    args = parser.parse_args()
    main(
        input_file=args.input_file,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
    )
