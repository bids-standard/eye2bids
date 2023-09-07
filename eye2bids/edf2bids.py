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

    # CONVERSION events
    input_file, metadata_file, output_dir = _check_inputs(
        input_file, metadata_file, output_dir
    )

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
    ManufacturersModelName = (
        " ".join([ml for ml in events if ml.startswith("** EYELINK")])
        .replace("** ", "")
        .replace("\n", "")
    )

    DeviceSerialNumber = (
        " ".join([sl for sl in events if sl.startswith("** SERIAL NUMBER:")])
        .replace("** SERIAL NUMBER: ", "")
        .replace("\n", "")
    )

    if df_ms[df_ms[3] == "VALIDATION"].empty is False:
        AverageCalibrationError = (
            np.array(df_ms[df_ms[3] == "VALIDATION"][[9]]).astype(float).tolist()
        )
    else:
        AverageCalibrationError = []

    CalibrationCount = len(df_ms_reduced[df_ms_reduced[3] == "CALIBRATION"])

    cal_pos = (
        np.array(
            df_ms_reduced[df_ms_reduced[2] == "VALIDATE"][8].str.split(",", expand=True)
        )
        .astype(int)
        .tolist()
    )
    cal_num = len(cal_pos) // CalibrationCount
    CalibrationPosition = []
    if len(cal_pos) != 0:
        CalibrationPosition.extend(
            cal_pos[i : i + cal_num] for i in range(0, len(cal_pos), cal_num)
        )
    CalibrationType = (
        df_ms_reduced[df_ms_reduced[3] == "CALIBRATION"]
        .iloc[0:1, 2:3]
        .to_string(header=False, index=False)
    )

    if len(cal_pos) != 0:
        cal_unit = (
            (df_ms_reduced[df_ms_reduced[2] == "VALIDATE"][[13]])
            .iloc[0:1, 0:1]
            .to_string(header=False, index=False)
        )
        if cal_unit == "cm":
            CalibrationUnit = "cm"
        elif cal_unit == "mm":
            CalibrationUnit = "mm"
        elif cal_unit == "pix.":
            CalibrationUnit = "pixel"
    else:
        CalibrationUnit = ""

    EyeTrackingMethod = (
        pd.DataFrame(
            " ".join([tm for tm in events if tm.startswith(">>>>>>>")])
            .replace(")", ",")
            .split(",")
        )
        .iloc[1:2]
        .to_string(header=False, index=False)
    )

    if df_ms[df_ms[3] == "VALIDATION"].empty is False:
        MaximalCalibrationError = (
            np.array(df_ms[df_ms[3] == "VALIDATION"][[11]]).astype(float).tolist()
        )
    else:
        MaximalCalibrationError = []

    PupilFitMethod = (
        (df_ms_reduced[df_ms_reduced[2] == "ELCL_PROC"])
        .iloc[0:1, 1:2]
        .to_string(header=False, index=False)
    )

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
        "AverageCalibrationError": AverageCalibrationError,
        "CalibrationCount": CalibrationCount,
        "CalibrationPosition": CalibrationPosition,
        "CalibrationUnit": CalibrationUnit,
        "CalibrationType": CalibrationType,
        "DeviceSerialNumber": DeviceSerialNumber,
        "EyeTrackingMethod": EyeTrackingMethod,
        "Manufacturer": "SR-Research",
        "ManufacturersModelName": ManufacturersModelName,
        "MaximalCalibrationError": MaximalCalibrationError,
        "PupilFitMethod": PupilFitMethod,
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


def _extract_TaskName(events):
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
