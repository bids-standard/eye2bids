import pandas as pd
import glob, os

plof_filepath = input('Enter the plof file path: ')
if os.path.exists(plof_filepath):
    print('The file exists')
else:
    raise FileNotFoundError('No such file or directory')


def _extract_recording_metadata_to_dataframe(plof_filepath):
    data_list = []

    with open(plof_filepath, 'r') as file:
        for line in file:
            if line.strip().startswith('[Recording]'):
            
                # Set the current header and data to be extracted
                current_header = next(file).strip()
                current_data = next(file).strip()
            
        # Create a DataFrame from the collected data
        if current_header is not None and current_data is not None:
            data_dict = dict(zip(current_header.split(), current_data.split('\t')))
            data_list.append(data_dict)

    df_recording = pd.DataFrame(data_list)

    return pd.DataFrame(data_list)

df_recording = _extract_recording_metadata_to_dataframe(plof_filepath)


def _extract_SoftwareVersion(df: pd.DataFrame) -> str:
    return (df.iloc[0, 11])

def _extract_StartTime(df: pd.DataFrame) -> str:
    return (df.iloc[0, 6])












# to json
    eyetrack_json = {
        "Manufacturer": "Tobii",
        "EnvironmentCoordinates": ,
        "EyeCameraSettings": ,
        "EyeTrackerDistance": ,
        "FeatureDetectionSettings": ,
        "GazeMappingSettings": ,
        "RawDataFilters": ,
        "SampleCoordinateSystem": ,
        "SampleCoordinateUnits": ,
        "ScreenAOIDefinition": ,
        "SoftwareVersion": _extract_SoftwareVersion(df_recording),
        "DeviceSerialNumber": ,
        "EyeTrackingMethod": ,
        "ManufacturersModelName": ,
        "AverageCalibrationError": ,
        "MaximalCalibrationError": ,
        "CalibrationCount": ,
        "CalibrationPosition": ,
        "CalibrationUnit": ,
        "CalibrationType": ,
        "PupilFitMethod": ,
        "RecordedEye": ,
        "SamplingFrequency": ,
        "StartTime": _extract_StartTime(df_recording),
        "StopTime": ,
    }
