"""Main module for conversion of edf to bids compliant files."""

from __future__ import annotations

import gzip
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rich.prompt import Prompt
from yaml.loader import SafeLoader

from eye2bids._base import BaseEventsJson, BasePhysioEventsJson, BasePhysioJson
from eye2bids._parser import global_parser
from eye2bids.logger import eye2bids_logger

e2b_log = eye2bids_logger()


def _check_inputs(
    input_file: str | Path | None = None,
    metadata_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    interactive: bool = False,
    force: bool = False,
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
        e2b_log.warning(
            """Load the metadata.yml file with the additional metadata.\n
            You can find a template in the eye2bids GitHub.\n
            This file must contain at least the additional REQUIRED metadata\n
            in the format specified in the BIDS specification.\n"""
        )
        metadata_file = Prompt.ask("Enter the file path to the metadata.yml file")

    if metadata_file in [None, ""]:
        if not force:
            e2b_log.error(
                """You didn't pass a metadata.yml file.
                As this file contains metadata
                which is REQUIRED for a valid BIDS dataset,
                the conversion process now stops.
                Please start again with a metadata.yml file
                or run eye2bids in --force mode.\n
                This will produce an invalid BIDS dataset.\n"""
            )
            raise SystemExit(1)
        else:
            e2b_log.warning(
                """You didn't pass a metadata.yml file.
                    Note that this will produce an invalid BIDS dataset.\n"""
            )

    checked_metadata_file = None
    if isinstance(metadata_file, str):
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


def _has_calibration(df: pd.DataFrame) -> bool:
    return not _calibrations(df).empty


def _extract_CalibrationType(df: pd.DataFrame) -> list[int]:
    return _calibrations(df).iloc[0:1, 2:3].to_string(header=False, index=False)


def _extract_CalibrationCount(df: pd.DataFrame, two_eyes: bool) -> int:
    return len(_calibrations(df)) // 2 if two_eyes else len(_calibrations(df))


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
    return ((_validations(df)[[11]]).astype(float)).to_numpy().tolist()


def _extract_AverageCalibrationError(df: pd.DataFrame) -> list[float]:
    return ((_validations(df)[[9]]).astype(float)).to_numpy().tolist()


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
    recorded_eye_map: dict[str, str | list[str]] = {
        "L": "Left",
        "R": "Right",
        "LR": ["Left", "Right"],
    }
    if eye in recorded_eye_map:
        return recorded_eye_map[eye]
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


def _extract_StartTime(events: list[str]) -> int:
    StartTime = (
        (pd.DataFrame([st.split() for st in events if st.startswith("START")])[1])
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


def _extract_StopTime(events: list[str]) -> int:
    StopTime = (
        (pd.DataFrame([so.split() for so in events if so.startswith("END")])[1])
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


def _load_asc_file(events_asc_file: str | Path) -> list[str]:
    with Path(events_asc_file).open() as f:
        return f.readlines()


def _load_asc_file_as_df(events_asc_file: str | Path) -> pd.DataFrame:
    # dataframe for events, all
    events = _load_asc_file(events_asc_file)
    return pd.DataFrame([ms.split() for ms in events if ms.startswith("MSG")])


def _load_asc_file_as_reduced_df(events_asc_file: str | Path) -> pd.DataFrame:
    """Reduce dataframe without MSG and sample columns."""
    df_ms = _load_asc_file_as_df(events_asc_file)
    return pd.DataFrame(df_ms.iloc[0:, 2:])


def _df_events_from_first_start(events: list[str]) -> pd.DataFrame:
    """Extract data starting from the first time START appears
    and including last time END appears.
    """
    start_index = next(
        i for i, line in enumerate(events) if re.match(r"START\s+.*", line)
    )
    end_index = next(
        i for i in range(len(events) - 1, -1, -1) if re.match(r"END\s+.*", events[i])
    )

    if end_index > start_index:
        data_lines = events[start_index : end_index + 1]
        return pd.DataFrame([line.strip().split("\t") for line in data_lines])
    else:
        return e2b_log.warning("No 'END' found after the selected 'START'.")


def _df_physioevents(events_from_start: pd.DataFrame) -> pd.DataFrame:
    events_from_start["Event_Letters"] = (
        events_from_start[0].str.extractall(r"([A-Za-z]+)").groupby(level=0).agg("".join)
    )
    events_from_start["Event_Numbers"] = events_from_start[0].str.extract(r"(\d+)")
    events_from_start[["msg_timestamp", "message"]] = events_from_start[1].str.split(
        n=1, expand=True
    )
    events_from_start["message"] = events_from_start["message"].astype(str)

    events_from_start["message"] = np.where(
        events_from_start["Event_Letters"] == "START",
        "START",
        np.where(
            events_from_start["Event_Letters"] == "END",
            "END",
            events_from_start.get("message", ""),
        ),
    )

    msg_mask = events_from_start["Event_Letters"].isin(["MSG", "START", "END"])
    events_from_start.loc[msg_mask, "Event_Numbers"] = events_from_start.loc[
        msg_mask, "msg_timestamp"
    ]

    physioevents_reordered = (
        pd.concat(
            [
                events_from_start["Event_Numbers"],
                events_from_start[2],
                events_from_start["Event_Letters"],
                events_from_start["message"],
            ],
            axis=1,
            ignore_index=True,
        )
        .replace({None: np.nan, "None": np.nan})
        .rename(columns={0: "timestamp", 1: "duration", 2: "trial_type", 3: "message"})
    )
    return physioevents_reordered


def _physioevents_for_eye(
    physioevents_reordered: pd.DataFrame, eye: str = "L"
) -> pd.DataFrame:
    physioevents_eye_list = [
        "MSG",
        f"EFIX{eye}",
        f"ESACC{eye}",
        f"EBLINK{eye}",
        "START",
        "END",
    ]

    physioevents = physioevents_reordered[
        physioevents_reordered["trial_type"].isin(physioevents_eye_list)
    ]

    physioevents["trial_type"] = physioevents["trial_type"].replace(
        {
            f"EFIX{eye}": "fixation",
            f"ESACC{eye}": "saccade",
            "MSG": np.nan,
            "START": np.nan,
            "END": np.nan,
            None: np.nan,
        }
    )

    physioevents["blink"] = 0
    last_non_na_trial_type = None

    for i in range(len(physioevents)):
        current_trial_type = physioevents.iloc[i]["trial_type"]
        if pd.notna(current_trial_type):
            if (
                current_trial_type == "saccade"
                and last_non_na_trial_type == f"EBLINK{eye}"
            ):
                physioevents.iloc[i, physioevents.columns.get_loc("blink")] = 1
            last_non_na_trial_type = current_trial_type

    physioevents.loc[physioevents["trial_type"].isna(), "blink"] = np.nan
    physioevents["blink"] = physioevents["blink"].astype("Int64")
    physioevents = physioevents[physioevents.trial_type != f"EBLINK{eye}"]

    physioevents["timestamp"] = physioevents["timestamp"].astype("Int64")
    physioevents["duration"] = pd.to_numeric(physioevents["duration"], errors="coerce")
    physioevents["duration"] = physioevents["duration"].astype("Int64")

    physioevents = physioevents[
        ["timestamp", "duration", "trial_type", "blink", "message"]
    ]
    return physioevents


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
        with Path(metadata_file).open() as f:
            metadata = yaml.load(f, Loader=SafeLoader)

    events = _load_asc_file(events_asc_file)
    df_ms = _load_asc_file_as_df(events_asc_file)
    df_ms_reduced = _load_asc_file_as_reduced_df(events_asc_file)

    base_json = BasePhysioJson(manufacturer="SR-Research", metadata=metadata)

    base_json.input_file = input_file
    base_json.has_validation = _has_validation(df_ms_reduced)
    base_json.two_eyes = _2eyesmode(df_ms_reduced)
    base_json.has_calibration = _has_calibration(df_ms_reduced)

    base_json["ManufacturersModelName"] = _extract_ManufacturersModelName(events)
    base_json["DeviceSerialNumber"] = _extract_DeviceSerialNumber(events)
    base_json["PupilFitMethod"] = _extract_PupilFitMethod(df_ms_reduced)
    base_json["SamplingFrequency"] = _extract_SamplingFrequency(df_ms_reduced)

    base_json["StartTime"] = _extract_StartTime(events)
    base_json["StopTime"] = _extract_StopTime(events)

    if base_json.has_calibration:
        base_json["EyeTrackingMethod"] = _extract_EyeTrackingMethod(events)

        base_json["CalibrationCount"] = _extract_CalibrationCount(
            df_ms_reduced, two_eyes=base_json.two_eyes
        )
        base_json["CalibrationType"] = _extract_CalibrationType(df_ms_reduced)

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
    e2b_log.info(f"file generated: {base_json.output_filename()}")
    if base_json.two_eyes:
        base_json.write(
            output_dir=output_dir, recording="eye2", extra_metadata=metadata_eye2
        )
        e2b_log.info(f"file generated: {base_json.output_filename()}")


def edf2bids(
    input_file: str | Path | None = None,
    metadata_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    interactive: bool = False,
    force: bool = False,
) -> None:
    """Convert edf to tsv + json."""
    if not _check_edf2asc_present():
        return

    input_file, metadata_file, output_dir = _check_inputs(
        input_file, metadata_file, output_dir, interactive, force
    )

    # CONVERSION events #
    events_asc_file = _convert_edf_to_asc_events(input_file)

    if not events_asc_file.exists():
        e2b_log.error(
            f"The following .edf input file could not be converted to .asc:{input_file}"
        )

    # SIDECARS #
    # %% physio.json
    generate_physio_json(input_file, metadata_file, output_dir, events_asc_file)
    # %% physioevents.json
    events = _load_asc_file(events_asc_file)

    df_ms_reduced = _load_asc_file_as_reduced_df(events_asc_file)

    physioevents_json = BasePhysioEventsJson()

    physioevents_json.input_file = input_file
    physioevents_json.two_eyes = _2eyesmode(df_ms_reduced)

    physioevents_json.write(output_dir=output_dir, recording="eye1")
    e2b_log.info(f"file generated: {physioevents_json.output_filename()}")
    if physioevents_json.two_eyes:
        physioevents_json.write(output_dir=output_dir, recording="eye2")
        e2b_log.info(f"file generated: {physioevents_json.output_filename()}")
    # %% events.json
    if metadata_file is None:
        metadata = {}
    else:
        with metadata_file.open() as f:
            metadata = yaml.load(f, Loader=SafeLoader)

    events_json = BaseEventsJson(metadata)

    events_json.input_file = input_file

    events_json["StimulusPresentation"]["ScreenResolution"] = _extract_ScreenResolution(
        df_ms_reduced
    )

    events_json.input_file = input_file

    events_json.write(output_dir=output_dir)
    e2b_log.info(f"file generated: {events_json.output_filename()}")

    # SAMPLES #
    # samples to dataframe
    samples_asc_file = _convert_edf_to_asc_samples(input_file)
    if not samples_asc_file.exists():
        e2b_log.error(
            f"The following .edf input file could not be converted to .asc:{input_file}"
        )

    samples = pd.read_csv(samples_asc_file, sep="\t", header=None)
    samples_eye1 = (
        pd.DataFrame(samples.iloc[:, [0, 1, 2, 3]])
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

    # MESSAGES AND PHYSIOEVENTS #
    # %%
    # Messages and events to dataframes

    events_from_start = _df_events_from_first_start(events)
    physioevents_reordered = _df_physioevents(events_from_start)
    physioevents_eye1 = _physioevents_for_eye(physioevents_reordered, eye="L")
    physioevents_eye2 = _physioevents_for_eye(physioevents_reordered, eye="R")

    # %%
    # Messages and events to physioevents.tsv.gz

    if not _2eyesmode(df_ms_reduced):
        output_eventsfilename_eye1 = generate_output_filename(
            output_dir=output_dir,
            input_file=input_file,
            suffix="_recording-eye1_physioevents",
            extension="tsv.gz",
        )
        if _extract_RecordedEye(df_ms_reduced) == "Left":
            data_to_save = physioevents_eye1
        elif _extract_RecordedEye(df_ms_reduced) == "Right":
            data_to_save = physioevents_eye2
        content = data_to_save.to_csv(sep="\t", index=False, na_rep="n/a", header=None)
        with gzip.open(output_eventsfilename_eye1, "wb") as f:
            f.write(content.encode())

        e2b_log.info(f"file generated: {output_eventsfilename_eye1}")

    else:
        output_eventsfilename_eye1 = generate_output_filename(
            output_dir=output_dir,
            input_file=input_file,
            suffix="_recording-eye1_physioevents",
            extension="tsv.gz",
        )
        content = physioevents_eye1.to_csv(
            sep="\t", index=False, na_rep="n/a", header=None
        )
        with gzip.open(output_eventsfilename_eye1, "wb") as f:
            f.write(content.encode())

        e2b_log.info(f"file generated: {output_eventsfilename_eye1}")

        output_eventsfilename_eye2 = generate_output_filename(
            output_dir=output_dir,
            input_file=input_file,
            suffix="_recording-eye2_physioevents",
            extension="tsv.gz",
        )
        content = physioevents_eye2.to_csv(
            sep="\t", index=False, na_rep="n/a", header=None
        )
        with gzip.open(output_eventsfilename_eye2, "wb") as f:
            f.write(content.encode())

        e2b_log.info(f"file generated: {output_eventsfilename_eye2}")


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
