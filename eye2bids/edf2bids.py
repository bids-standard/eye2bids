"""Main module for conversion of edf to bids compliant files."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rich.prompt import Prompt
from yaml.loader import SafeLoader

from eye2bids._parser import global_parser
from eye2bids.logger import eye2bids_logger

e2b_log = eye2bids_logger()


def _check_inputs(
    input_file: str | Path | None = None,
    metadata_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    interactive: bool = False,
) -> tuple[Path, Path | None, Path]:
    """Check if inputs are valid."""
    if input_file is None:
        if interactive:
            input_file = Prompt.ask("Enter the edf file path")
        else:
            raise FileNotFoundError("No input file specified")

    if isinstance(input_file, str):
        cheked_input_file = Path(input_file)
    elif isinstance(input_file, Path):
        cheked_input_file = input_file

    if cheked_input_file.exists():
        e2b_log.info(f"input file found: {cheked_input_file}")
    else:
        raise FileNotFoundError(f"No such input file: {cheked_input_file}")

    if metadata_file in [None, ""] and interactive:
        e2b_log.info(
            """Load the metadata.yml file with the additional metadata.\n
            This file must contain at least the additional REQUIRED metadata
            in the format specified in the BIDS specification.\n"""
        )
        metadata_file = Prompt.ask("Enter the file path to the metadata.yml file")

    if metadata_file in ["", None]:
        checked_metadata_file = None
    elif isinstance(metadata_file, str):
        checked_metadata_file = Path(metadata_file)
    elif isinstance(metadata_file, Path):
        checked_metadata_file = metadata_file

    if isinstance(checked_metadata_file, Path):
        if not checked_metadata_file.exists():
            raise FileNotFoundError(f"No such metadata file: {checked_metadata_file}")
        if checked_metadata_file.is_file():
            e2b_log.info(f"metadata file found: {checked_metadata_file}")
        elif checked_metadata_file.is_dir():
            raise IsADirectoryError(
                f"metadata file is a directory: {checked_metadata_file}"
            )

    return cheked_input_file, checked_metadata_file, _check_output_dir(output_dir)


def _check_output_dir(output_dir: str | Path | None = None) -> Path:
    """Check if output directory is valid."""
    if output_dir is None:
        output_dir = input("Enter the output directory: ")
    if isinstance(output_dir, str):
        checked_output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        checked_output_dir = output_dir

    if not checked_output_dir.exists():
        checked_output_dir.mkdir(parents=True, exist_ok=True)

    return checked_output_dir


def _check_edf2asc_present() -> bool:
    """Check if edf2asc is present in the path."""
    try:
        subprocess.run(["edf2asc"])
        return True
    except FileNotFoundError:
        e2b_log.error(
            """edf2asc not found in path.
Make sure to install it from https://www.sr-research.com/."""
        )
        return False


def _convert_edf_to_asc_events(input_file: str | Path) -> Path:
    """Convert edf to asc - events."""
    if isinstance(input_file, str):
        input_file = Path(input_file)
    events_asc_file = (input_file.parent) / (input_file.stem + "_events")
    subprocess.run(["edf2asc", "-y", "-e", input_file, "-o", events_asc_file])
    return Path(events_asc_file).with_suffix(".asc")


def _convert_edf_to_asc_samples(input_file: str | Path) -> Path:
    """Convert edf to asc - samples."""
    if isinstance(input_file, str):
        input_file = Path(input_file)
    samples_asc_file = (input_file.parent) / (input_file.stem + "_samples")
    subprocess.run(["edf2asc", "-y", "-s", input_file, "-o", samples_asc_file])
    return Path(samples_asc_file).with_suffix(".asc")

def _2eyesmode(df: pd.DataFrame) -> bool:
    eye = df[df[2] == "RECCFG"].iloc[0:1, 5:6].to_string(header=False, index=False)
    if eye == "LR":
        two_eyes = True
    else:
        two_eyes = False
    return two_eyes

def _calibrations(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[3] == "CALIBRATION"]

def _extract_CalibrationType(df: pd.DataFrame) -> list[int]:
    return _calibrations(df).iloc[0:1, 2:3].to_string(header=False, index=False)


def _extract_CalibrationCount(df: pd.DataFrame) -> int:
    if _2eyesmode(df) == True:
        return len(_calibrations(df)) // 2
    return len(_calibrations(df))


def _get_calibration_positions(df: pd.DataFrame) -> list[int]:
    if _2eyesmode(df) == True:
        return (
            np.array(df[df[2] == "VALIDATE"][8].str.split(",", expand=True))
            .astype(int)
            .tolist()
        )[::2]
    return (
        np.array(df[df[2] == "VALIDATE"][8].str.split(",", expand=True))
        .astype(int)
        .tolist()
    )


def _extract_CalibrationPosition(df: pd.DataFrame) -> list[list[int]]:
    cal_pos = _get_calibration_positions(df)
    cal_num = len(cal_pos) // _extract_CalibrationCount(df)

    CalibrationPosition: list[list[int]] = []

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
    if cal_unit == "pix.":
        return "pixel"
    elif cal_unit in ["cm", "mm"]:
        return cal_unit
    return ""


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


def _validations(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[3] == "VALIDATION"]


def _has_validation(df: pd.DataFrame) -> bool:
    return not _validations(df).empty


def _extract_MaximalCalibrationError(df: pd.DataFrame) -> list[float]:
    if not _has_validation(df):
        return []
    return np.array(_validations(df)[[11]]).astype(float).tolist()


def _extract_AverageCalibrationError(df: pd.DataFrame) -> list[float]:
    if not _has_validation(df):
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
        return ["Left", "Right"]
    return "" 


def _extract_ScreenResolution(df: pd.DataFrame) -> list[int]:
    list_res = (
        (df[df[2] == "GAZE_COORDS"])
        .iloc[0:1, 3:5]
        .to_string(header=False, index=False)
        .replace(".00", "")
        .split(" ")
    )
    return [eval(i) for i in list_res]


def _extract_TaskName(events: list[str]) -> str:
    return (
        " ".join([ts for ts in events if ts.startswith("** RECORDED BY")])
        .replace("** RECORDED BY ", "")
        .replace("\n", "")
    )


def _extract_StartTime(events: list[str]) -> int:
    StartTime = (
        np.array(pd.DataFrame([st.split() for st in events if st.startswith("START")])[1])
        .astype(int)
        .tolist()
    )
    if len(StartTime) > 1:
        e2b_log.info(
            """Your input file contains multiple start times.\n
             As this is not seen as good practice in eyetracking experiments, \n
             only the first start time will be kept for the metadata file. \n
             Please consider changing your code accordingly
             for future eyetracking experiments.\n"""
        )
        return StartTime[0]
    return StartTime


def _extract_StopTime(events: list[str]) -> int:
    StopTime = (
        np.array(pd.DataFrame([so.split() for so in events if so.startswith("END")])[1])
        .astype(int)
        .tolist()
    )
    if len(StopTime) > 1:
        e2b_log.info(
            """Your input file contains multiple stop times.\n
             As this is not seen as good practice in eyetracking experiments, \n
             only the last stop time will be kept for the metadata file. \n
             Please consider changing your code accordingly
             for future eyetracking experiments.\n"""
        )
        return StopTime[-1]
    return StopTime


def _load_asc_file(events_asc_file: str | Path) -> list[str]:
    with open(events_asc_file) as f:
        return f.readlines()


def _load_asc_file_as_df(events_asc_file: str | Path) -> pd.DataFrame:
    # dataframe for events, all
    events = _load_asc_file(events_asc_file)
    return pd.DataFrame([ms.split() for ms in events if ms.startswith("MSG")])


def _load_asc_file_as_reduced_df(events_asc_file: str | Path) -> pd.DataFrame:
    # reduced dataframe without MSG and sample columns
    df_ms = _load_asc_file_as_df(events_asc_file)
    return pd.DataFrame(df_ms.iloc[0:, 2:])

def edf2bids(
    input_file: str | Path | None = None,
    metadata_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    interactive: bool = False,
) -> None:
    """Convert edf to tsv + json."""
    if not _check_edf2asc_present():
        return

    input_file, metadata_file, output_dir = _check_inputs(
        input_file, metadata_file, output_dir, interactive
    )

    # CONVERSION events
    events_asc_file = _convert_edf_to_asc_events(input_file)

    if not events_asc_file.exists():
        e2b_log.error(
            "The following .edf input file could not be converted to .asc:"
            f"{input_file}"
        )

    events = _load_asc_file(events_asc_file)
    df_ms = _load_asc_file_as_df(events_asc_file)
    df_ms_reduced = _load_asc_file_as_reduced_df(events_asc_file)

    if metadata_file is None:
        metadata = {}
    else:
        with open(metadata_file) as f:
            metadata = yaml.load(f, Loader=SafeLoader)


    # eye-physio.json Metadata
    base_json = {
            "Columns": [ "x_coordinate", "y_coordinate", "pupil_size", "timestamp"],
            "x_coordinate": {
                "Description": "Gaze position x-coordinate of the recorded eye, in the coordinate units specified in the corresponding metadata sidecar.",
                "Units": "a.u."
            },
            "y_coordinate": {
                "Description": "Gaze position y-coordinate of the recorded eye, in the coordinate units specified in the corresponding metadata sidecar.",
                "Units": "a.u."
            },
            "pupil_size": {
                "Description": "Pupil area of the recorded eye as calculated by the eye-tracker in arbitrary units (see EyeLink's documentation for conversion).",
                 "Units": "a.u."
             },
             "timestamp": {
                "Description": "Timestamp issued by the eye-tracker indexing the continuous recordings corresponding to the sampled eye."
            },
            "Manufacturer": "SR-Research",
            "ManufacturersModelName": _extract_ManufacturersModelName(events),
            "DeviceSerialNumber": _extract_DeviceSerialNumber(events),
            "EnvironmentCoordinates": metadata.get("EnvironmentCoordinates"),
            "SoftwareVersion": metadata.get("SoftwareVersion"),
            "EyeCameraSettings": metadata.get("EyeCameraSettings"),
            "EyeTrackerDistance": metadata.get("EyeTrackerDistance"),
            "FeatureDetectionSettings": metadata.get("FeatureDetectionSettings"),
            "GazeMappingSettings": metadata.get("GazeMappingSettings"),
            "RawDataFilters": metadata.get("RawDataFilters"),
            "SampleCoordinateSystem": metadata.get("SampleCoordinateSystem"),
            "SampleCoordinateUnits": metadata.get("SampleCoordinateUnits"),
            "ScreenAOIDefinition": metadata.get("ScreenAOIDefinition"),
            "EyeTrackingMethod": _extract_EyeTrackingMethod(events),
            "PupilFitMethod": _extract_PupilFitMethod(df_ms_reduced),
            "SamplingFrequency": _extract_SamplingFrequency(df_ms_reduced),
            "StartTime": _extract_StartTime(events),
            "StopTime": _extract_StopTime(events),
            "CalibrationUnit": _extract_CalibrationUnit(df_ms_reduced),
            "CalibrationType": _extract_CalibrationType(df_ms_reduced),
            "CalibrationCount": _extract_CalibrationCount(df_ms_reduced),
            "CalibrationPosition": _extract_CalibrationPosition(df_ms_reduced)
        }
    
    if _2eyesmode(df_ms_reduced) == True:
        metadata_eye1 = {
                "AverageCalibrationError": (_extract_AverageCalibrationError(df_ms)[0::2]),
                "MaximalCalibrationError": (_extract_MaximalCalibrationError(df_ms)[0::2]),
                "RecordedEye": (_extract_RecordedEye(df_ms_reduced)[0])
            }
    
        metadata_eye2 = {
                "AverageCalibrationError": (_extract_AverageCalibrationError(df_ms)[1::2]),
                "MaximalCalibrationError": (_extract_MaximalCalibrationError(df_ms)[1::2]),
                "RecordedEye": (_extract_RecordedEye(df_ms_reduced)[1])
        }
    else:
        metadata_eye1 = {
                "AverageCalibrationError": (_extract_AverageCalibrationError(df_ms)[0::2]),
                "MaximalCalibrationError": (_extract_MaximalCalibrationError(df_ms)[0::2]),
                "RecordedEye": (_extract_RecordedEye(df_ms_reduced))
            }
        
    json_eye1 = dict(base_json, **metadata_eye1) 
    if _2eyesmode(df_ms_reduced) == True:
        json_eye2 = dict(base_json, **metadata_eye2)

    # to json
    
    output_filename_eye1 = generate_output_filename(
                output_dir=output_dir, input_file=input_file, suffix="_recording-eye1_physio", extension="json"
            )
    with open(output_filename_eye1, "w") as outfile:
            json.dump(json_eye1, outfile, indent=4)

    e2b_log.info(f"file generated: {output_filename_eye1}")

    if _2eyesmode(df_ms_reduced) == True:
        output_filename_eye2 = generate_output_filename(
                output_dir=output_dir, input_file=input_file, suffix="_recording-eye2_physio", extension="json"
            )
        with open(output_filename_eye2, "w") as outfile:
            json.dump(json_eye2, outfile, indent=4)
        
        e2b_log.info(f"file generated: {output_filename_eye2}")

    # physioevents.json Metadata

    events_json = {
        "Columns": ["onset", "duration", "trial_type", "blink", "message"],
        "Description": "Messages logged by the measurement device",
        "ForeignIndexColumn": "timestamp",
        "blink": {
            "Description": "One indicates if the eye was closed, zero if open."
        },
        "message": {
            "Description": "String messages logged by the eye-tracker."
        },
        "trial_type": {
        "Description": "Event type as identified by the eye-tracker's model (either 'n/a' if not applicabble, 'fixation', or 'saccade')."
        },
        "TaskName": _extract_TaskName(events), 
        "InstitutionAddress": metadata.get("InstitutionAddress"),
        "InstitutionName": metadata.get("InstitutionName"),
        "StimulusPresentation": {
            "ScreenDistance": metadata.get("ScreenDistance"),
            "ScreenRefreshRate": metadata.get("ScreenRefreshRate"),
            "ScreenSize": metadata.get("ScreenSize"),
            "ScreenResolution": _extract_ScreenResolution(df_ms_reduced),
        }
    }

    
    output_filename_eye1 = generate_output_filename(
            output_dir=output_dir, input_file=input_file, suffix="_recording-eye1_physioevents", extension="json"
        )
    with open(output_filename_eye1, "w") as outfile:
        json.dump(events_json, outfile, indent=4)

    e2b_log.info(f"file generated: {output_filename_eye1}")

    if _2eyesmode(df_ms_reduced) == True:

        output_filename_eye2 = generate_output_filename(
                output_dir=output_dir, input_file=input_file, suffix="_recording-eye2_physioevents", extension="json"
            )
        with open(output_filename_eye2, "w") as outfile:
            json.dump(events_json, outfile, indent=4)
        
        e2b_log.info(f"file generated: {output_filename_eye2}")

    # Samples to dataframe

    samples_asc_file = _convert_edf_to_asc_samples(input_file)
    if not samples_asc_file.exists():
        e2b_log.error(
            "The following .edf input file could not be converted to .asc:"
            f"{input_file}"
        )

    samples = pd.read_csv(samples_asc_file, sep="\t", header=None)
    samples_eye1 = pd.DataFrame(samples.iloc[:, [2, 1, 3, 0]]).map(lambda x: x.strip() if isinstance(x, str) else x).replace(".", np.nan, regex=False)
    
    if _2eyesmode(df_ms_reduced) == True:
        samples_eye2 = pd.DataFrame(samples.iloc[:, [4, 5, 6, 0]])

    # Samples to eye_physio.tsv.gz

    output_filename_eye1 = generate_output_filename(
        output_dir=output_dir, input_file=input_file, suffix="_recording-eye1_physio", extension="tsv.gz"
        )
    content = samples_eye1.to_csv(sep="\t", index=False, na_rep="n/a", header=None)
    with gzip.open(output_filename_eye1, "wb") as f:
        f.write(content.encode())

    e2b_log.info(f"file generated: {output_filename_eye1}")

    if _2eyesmode(df_ms_reduced) == True:

        output_filename_eye2 = generate_output_filename(
                output_dir=output_dir, input_file=input_file, suffix="_recording-eye2_physio", extension="tsv.gz"
            )
        content = samples_eye2.to_csv(sep="\t", index=False, na_rep="n/a", header=None)
        with gzip.open(output_filename_eye2, "wb") as f:
            f.write(content.encode())
        
        e2b_log.info(f"file generated: {output_filename_eye2}")


    # Messages and events to physioevents.tsv.gz - tbc



def generate_output_filename(
    output_dir: Path, input_file: Path, suffix: str, extension: str
) -> Path:
    """Generate output filename."""
    filename = Path(input_file).stem
    if filename.endswith(suffix):
        suffix = ""
    return output_dir / f"{filename}{suffix}.{extension}"


if __name__ == "__main__":
    parser = global_parser()
    args = parser.parse_args()
    edf2bids(
        input_file=args.input_file,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
    )
