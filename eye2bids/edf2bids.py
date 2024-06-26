"""Main module for conversion of edf to bids compliant files."""

from __future__ import annotations

import gzip
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rich.prompt import Prompt
from yaml.loader import SafeLoader

from eye2bids._base import BasePhysioEventsJson, BasePhysioJson
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
        output_dir = Path()
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
    two_eyes = eye == "LR"
    return two_eyes


def _calibrations(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[3] == "CALIBRATION"]


def _extract_CalibrationType(df: pd.DataFrame) -> list[int]:
    return _calibrations(df).iloc[0:1, 2:3].to_string(header=False, index=False)


def _extract_CalibrationCount(df: pd.DataFrame, two_eyes: bool) -> int:
    return len(_calibrations(df)) // 2 if two_eyes else len(_calibrations(df))


def _extract_CalibrationPosition(df: pd.DataFrame) -> list[list[list[int]]]:

    calibration_df = df[df[2] == "VALIDATE"]
    calibration_df[5] = pd.to_numeric(calibration_df[5], errors="coerce")

    if _2eyesmode(df):
        # drop duplicated calibration position
        # because they will be the same for both eyes
        calibration_df = calibration_df[calibration_df[6] == "LEFT"]

    nb_calibration_postions = calibration_df[5].max() + 1

    # initialize
    CalibrationPosition: Any = [[[]] * nb_calibration_postions]

    for i_pos in range(nb_calibration_postions):

        results_for_this_position = calibration_df[calibration_df[5] == i_pos]

        for i, calibration in enumerate(results_for_this_position.iterrows()):
            values = calibration[1][8].split(",")

            if len(CalibrationPosition) < i + 1:
                CalibrationPosition.append([[]] * nb_calibration_postions)

            CalibrationPosition[i][i_pos] = [int(x) for x in values]

    return CalibrationPosition


def _extract_CalibrationUnit(df: pd.DataFrame) -> str:
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
    return np.array(_validations(df)[[11]]).astype(float).tolist()


def _extract_AverageCalibrationError(df: pd.DataFrame) -> list[float]:
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


def _extract_RecordedEye(df: pd.DataFrame) -> str | list[str]:
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


def generate_physio_json(
    input_file: Path,
    metadata_file: str | Path | None,
    output_dir: Path,
    events_asc_file: Path,
) -> None:
    """Generate the _physio.json."""
    if metadata_file is None:
        metadata = {}
    else:
        with open(metadata_file) as f:
            metadata = yaml.load(f, Loader=SafeLoader)

    events = _load_asc_file(events_asc_file)
    df_ms = _load_asc_file_as_df(events_asc_file)
    df_ms_reduced = _load_asc_file_as_reduced_df(events_asc_file)

    base_json = BasePhysioJson(manufacturer="SR-Research", metadata=metadata)

    base_json.input_file = input_file
    base_json.has_validation = _has_validation(df_ms_reduced)
    base_json.two_eyes = _2eyesmode(df_ms_reduced)

    base_json["ManufacturersModelName"] = _extract_ManufacturersModelName(events)
    base_json["DeviceSerialNumber"] = _extract_DeviceSerialNumber(events)
    base_json["EyeTrackingMethod"] = _extract_EyeTrackingMethod(events)
    base_json["PupilFitMethod"] = _extract_PupilFitMethod(df_ms_reduced)
    base_json["SamplingFrequency"] = _extract_SamplingFrequency(df_ms_reduced)

    base_json["StartTime"] = _extract_StartTime(events)
    base_json["StopTime"] = _extract_StopTime(events)

    if base_json.two_eyes:
        metadata_eye1: dict[str, str | list[str] | list[float]] = {
            "RecordedEye": (_extract_RecordedEye(df_ms_reduced)[0]),
        }
        metadata_eye2: dict[str, str | list[str] | list[float]] = {
            "RecordedEye": (_extract_RecordedEye(df_ms_reduced)[1]),
        }
    else:
        metadata_eye1 = {
            "RecordedEye": (_extract_RecordedEye(df_ms_reduced)),
        }

    if base_json.has_validation:

        if CalibrationPosition := _extract_CalibrationPosition(df_ms_reduced):
            base_json["CalibrationCount"] = _extract_CalibrationCount(
                df_ms_reduced, two_eyes=base_json.two_eyes
            )
            base_json["CalibrationUnit"] = _extract_CalibrationUnit(df_ms_reduced)
            base_json["CalibrationType"] = _extract_CalibrationType(df_ms_reduced)

            base_json["CalibrationPosition"] = CalibrationPosition
            if base_json["CalibrationCount"] == 1:
                base_json["CalibrationPosition"] = CalibrationPosition[0]

        metadata_eye1["AverageCalibrationError"] = _extract_AverageCalibrationError(
            df_ms
        )[::2]
        metadata_eye1["MaximalCalibrationError"] = _extract_MaximalCalibrationError(
            df_ms
        )[::2]

        if base_json.two_eyes:
            metadata_eye2["AverageCalibrationError"] = _extract_AverageCalibrationError(
                df_ms
            )[1::2]
            metadata_eye2["MaximalCalibrationError"] = _extract_MaximalCalibrationError(
                df_ms
            )[1::2]

    base_json.write(output_dir=output_dir, recording="eye1", extra_metadata=metadata_eye1)
    if base_json.two_eyes:
        base_json.write(
            output_dir=output_dir, recording="eye2", extra_metadata=metadata_eye2
        )


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

    # %% Sidecar eye-physio.json
    generate_physio_json(input_file, metadata_file, output_dir, events_asc_file)

    # %% physioevents.json Metadata
    events = _load_asc_file(events_asc_file)

    df_ms_reduced = _load_asc_file_as_reduced_df(events_asc_file)

    if metadata_file is None:
        metadata = {}
    else:
        with open(metadata_file) as f:
            metadata = yaml.load(f, Loader=SafeLoader)

    events_json = BasePhysioEventsJson(metadata)

    events_json.input_file = input_file
    events_json.two_eyes = _2eyesmode(df_ms_reduced)

    events_json["TaskName"] = _extract_TaskName(events)
    events_json["StimulusPresentation"]["ScreenResolution"] = _extract_ScreenResolution(
        df_ms_reduced
    )

    events_json.write(output_dir=output_dir, recording="eye1")
    if events_json.two_eyes:
        events_json.write(output_dir=output_dir, recording="eye2")

    #  %%
    # Samples to dataframe
    samples_asc_file = _convert_edf_to_asc_samples(input_file)
    if not samples_asc_file.exists():
        e2b_log.error(
            "The following .edf input file could not be converted to .asc:"
            f"{input_file}"
        )

    samples = pd.read_csv(samples_asc_file, sep="\t", header=None)
    samples_eye1 = (
        pd.DataFrame(samples.iloc[:, 0:4])
        .map(lambda x: x.strip() if isinstance(x, str) else x)
        .replace(".", np.nan, regex=False)
    )

    if _2eyesmode(df_ms_reduced):
        samples_eye2 = pd.DataFrame(samples.iloc[:, [0, 4, 5, 6]])

    # %%
    # Samples to eye_physio.tsv.gz

    output_filename_eye1 = generate_output_filename(
        output_dir=output_dir,
        input_file=input_file,
        suffix="_recording-eye1_physio",
        extension="tsv.gz",
    )
    content = samples_eye1.to_csv(sep="\t", index=False, na_rep="n/a", header=None)
    with gzip.open(output_filename_eye1, "wb") as f:
        f.write(content.encode())

    e2b_log.info(f"file generated: {output_filename_eye1}")

    if _2eyesmode(df_ms_reduced):

        output_filename_eye2 = generate_output_filename(
            output_dir=output_dir,
            input_file=input_file,
            suffix="_recording-eye2_physio",
            extension="tsv.gz",
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
