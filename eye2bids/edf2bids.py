"""Main module for conversion of edf to bids compliant files."""
import json
import os
import subprocess

import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader


def main():
    """Convert edf to tsv + json."""
    # CONVERSION events

    input_file = input("Enter the edf file path: ")
    if os.path.exists(input_file):
        print("The file exists")
    else:
        raise FileNotFoundError("No such file or directory")

    subprocess.run(["edf2asc", "-y", "-e", input_file, input_file + "_events"])
    asc_file = input_file + "_events.asc"

    with open(asc_file) as f:
        events = f.readlines()

    # Read variables from the additional metadata txt file
    print(
        """Load the metadata.yml file with the additional metadata.\n
This file must contain at least the additional REQUIRED metadata
in the format specified in the BIDS specification.\n
Please enter the required metadata manually
before loading the file in a next step."""
    )

    metadata_file = input("Enter the file path to the metadata.yml file: ")

    if os.path.exists(metadata_file):
        print("The file exists")
    else:
        raise FileNotFoundError("No such file or directory")

    with open(metadata_file) as f:
        metadata = yaml.load(f, Loader=SafeLoader)

    SampleCoordinateUnits = metadata["SampleCoordinateUnits"]
    SampleCoordinateSystem = metadata["SampleCoordinateSystem"]
    EnvironmentCoordinates = metadata["EnvironmentCoordinates"]
    ScreenDistance = metadata["ScreenDistance"]
    ScreenRefreshRate = metadata["ScreenRefreshRate"]
    ScreenSize = metadata["ScreenSize"]
    InstitutionName = metadata["InstitutionName"]
    InstitutionAddress = metadata["InstitutionAddress"]
    SoftwareVersion = metadata["SoftwareVersion"]
    ScreenAOIDefinition = metadata["ScreenAOIDefinition"]
    EyeCameraSettings = metadata["EyeCameraSettings"]
    EyeTrackerDistance = metadata["EyeTrackerDistance"]
    FeatureDetectionSettings = metadata["FeatureDetectionSettings"]
    GazeMappingSettings = metadata["GazeMappingSettings"]
    RawDataFilters = metadata["RawDataFilters"]

    # Prepare asc file
    # dataframe for events, all
    df_ms = pd.DataFrame([ms.split() for ms in events if ms.startswith("MSG")])

    # reduced dataframe without MSG and sample columns
    df_ms_reduced = pd.DataFrame(df_ms.iloc[0:, 2:])

    # Events.json Metadata

    ScreenResolution = list(
        map(
            int,
            (
                df_ms_reduced[df_ms_reduced[2] == "DISPLAY_COORDS"]
                .iloc[0:1, 3:5]
                .to_string(header=False, index=False)
            ).split(" "),
        )
    )

    TaskName = (
        " ".join([ts for ts in events if ts.startswith("** RECORDED BY")])
        .replace("** RECORDED BY ", "")
        .replace("\n", "")
    )

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

    SamplingFrequency = int(
        df_ms_reduced[df_ms_reduced[2] == "RECCFG"]
        .iloc[0:1, 2:3]
        .to_string(header=False, index=False)
    )

    eye = (
        df_ms_reduced[df_ms_reduced[2] == "RECCFG"]
        .iloc[0:1, 5:6]
        .to_string(header=False, index=False)
    )
    if eye == "L":
        RecordedEye = eye.replace("L", "Left")
    elif eye == "R":
        RecordedEye = eye.replace("R", "Right")
    elif eye == "LR":
        RecordedEye = eye.replace("LR", "Both")

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
    cal_num = int(len(cal_pos) / CalibrationCount)
    CalibrationPosition = list()
    if len(cal_pos) != 0:
        for i in range(0, len(cal_pos), cal_num):
            CalibrationPosition.append(cal_pos[i : i + cal_num])

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
        if cal_unit == "pix.":
            CalibrationUnit = "pixel"
        elif cal_unit == "mm":
            CalibrationUnit = "mm"
        elif cal_unit == "cm":
            CalibrationUnit = "cm"
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

    StartTime = (
        np.array(pd.DataFrame([st.split() for st in events if st.startswith("START")])[1])
        .astype(int)
        .tolist()
    )  # TODO:figure out if this is actually the StartTime meant by the specification

    StopTime = (
        np.array(pd.DataFrame([so.split() for so in events if so.startswith("END")])[1])
        .astype(int)
        .tolist()
    )  # TODO:figure out if this is actually the StopTime meant by the specification

    # to json

    out_filepath = input("Enter the output directory: ")

    eyetrack_json = {
        "Manufacturer": "SR-Research",
        "ManufacturersModelName": ManufacturersModelName,
        "DeviceSerialNumber": DeviceSerialNumber,
        "SoftwareVersion": SoftwareVersion,
        "SamplingFrequency": SamplingFrequency,
        "SampleCoordinateUnits": SampleCoordinateUnits,
        "SampleCoordinateSystem": SampleCoordinateSystem,
        "EnvironmentCoordinates": EnvironmentCoordinates,
        "RecordedEye": RecordedEye,
        "ScreenAOIDefinition": ScreenAOIDefinition,
        "AverageCalibrationError": AverageCalibrationError,
        "CalibrationCount": CalibrationCount,
        "CalibrationType": CalibrationType,
        "CalibrationUnit": CalibrationUnit,
        "EyeCameraSettings": EyeCameraSettings,
        "EyeTrackerDistance": EyeTrackerDistance,
        "CalibrationPosition": CalibrationPosition,
        "EyeTrackingMethod": EyeTrackingMethod,
        "FeatureDetectionSettings": FeatureDetectionSettings,
        "GazeMappingSettings": GazeMappingSettings,
        "MaximalCalibrationError": MaximalCalibrationError,
        "PupilFitMethod": PupilFitMethod,
        "RawDataFilters": RawDataFilters,
        "StartTime": StartTime,
        "StopTime": StopTime,
    }

    with open(out_filepath + "eyetrack.json", "w") as outfile:
        json.dump(eyetrack_json, outfile, indent=15)

    events_json = {
        "TaskName": TaskName,
        "InstitutionName": InstitutionName,
        "InstitutionAddress": InstitutionAddress,
        "StimulusPresentation": {
            "ScreenDistance": ScreenDistance,
            "ScreenRefreshRate": ScreenRefreshRate,
            "ScreenResolution": ScreenResolution,
            "ScreenSize": ScreenSize,
        },
    }

    with open(out_filepath + "events.json", "w") as outfile:
        json.dump(events_json, outfile, indent=9)


if __name__ == "__main__":
    main()
