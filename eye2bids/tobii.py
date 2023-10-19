import pandas as pd
import glob, os
import json


# read in file

plof_filepath = input('Enter the plof file path: ')
if os.path.exists(plof_filepath):
    print('The file exists')
else:
    raise FileNotFoundError('No such file or directory')

# functions for creating necessary dataframes

def _extract_recording_metadata_to_dataframe(plof_filepath):
    data_list = []

    with open(plof_filepath, 'r') as file:
        for line in file:
            if line.strip().startswith('[Recording]'):
            
                current_header = next(file).strip()
                current_data = next(file).strip()
            
        if current_header is not None and current_data is not None:
            data_dict = dict(zip(current_header.split(), current_data.split('\t')))
            data_list.append(data_dict)

    return pd.DataFrame(data_list)

def _extract_data_section_to_dataframe(plof_filepath):
    lines = []
    header = None  

    with open(plof_filepath, 'r') as file:
        inside_data_section = False
        for line in file:
            line = line.strip()  
            if line == '[Data]':
                inside_data_section = True
                continue  
            if inside_data_section:
                if not header:
                    header = line.split('\t')  
                else:
                    if not line:  
                        break
                    columns = line.split('\t')
                    lines.append(columns)

    return pd.DataFrame(lines, columns=header)

def _extract_event_section_to_dataframe(plof_filepath):
    lines = []
    header = None  

    with open(plof_filepath, 'r') as file:
        inside_data_section = False
        for line in file:
            line = line.strip()  
            if line == '[Event]':
                inside_data_section = True
                continue  
            if inside_data_section:
                if not header:
                    header = line.split('\t')  
                else:
                    if not line:  
                        break
                    columns = line.split('\t')
                    lines.append(columns)

    return pd.DataFrame(lines, columns=header)

# dataframes

df_recording = _extract_recording_metadata_to_dataframe(plof_filepath)
df_data = _extract_data_section_to_dataframe(plof_filepath)
df_events = _extract_event_section_to_dataframe(plof_filepath)


# functions for fields

def _extract_SoftwareVersion(df: pd.DataFrame) -> str:
    return df.iloc[0, 11]

def _extract_StartTime(df: pd.DataFrame) -> int:
    return int(df.iloc[0, 0])

def _extract_StopTime(df: pd.DataFrame) -> int:
    return int(df_data.iloc[(len(df_data)-1), 0])

def _extract_ScreenResolution(df: pd.DataFrame) -> list[int]:
    width = int(df.iloc[0, 13])   
    height = int(df.iloc[0, 12])
    return list[width, height]

def _extract_TaskName(df: pd.DataFrame) -> str:
    return df.iloc[0, 0]




# to json
    eyetrack_json = {
        "Manufacturer": "Tobii",
        "EnvironmentCoordinates":,
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
        "StartTime": _extract_StartTime(df_data),
        "StopTime": _extract_StopTime(df_data),
    }

events_json={
   "TaskName": _extract_TaskName(df_recording),
   "InstitutionName": ,
   "InstitutionAddress": ,
   "StimulusPresentation": {
       "ScreenDistance": ,
       "ScreenRefreshRate": ,
       "ScreenResolution": _extract_ScreenResolution(df_recording),
       "ScreenSize": ,
   }
}