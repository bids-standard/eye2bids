# Using eye2bids

!!! info "Supported Eyetrackers"

    As of today, only edf files by EyeLink eye trackers are supported.

## Preparations

- in- and output folder in BIDS format, for example:

```bash
.
├─sourcedata
│ └─ sub-01
│    └─ beh
│ 		└─ sub-01_task-FreeView.edf
├─sub-01
│ └─ beh # empty
...
```

- create a `manual-metadata.tsv` table with the following contents:[^1]

??? info "Why?"

    Most of the metadata will be extracted by `eye2bids` from the given edf file. However, not all the metadata required in the BIDS specification is saved in the edf file. Thus, the `manual-metadata.tsv` will be read by `eye2bids` additionally to the edf files to produce a valid BIDS dataset. Given that these metadata should ideally be the same for one dataset, you only need to enter it once in the `manual-metadata.tsv`.


```py
# DO NOT COPY THIS BLOCK! It is not properly tab separated.

SampleCoordinateUnits           pixel
SampleCoordinateSystem          gaze-on-screen
EnvironmentCoordinates          top-left
ScreenDistance                  60
ScreenRefreshRate               120
ScreenSize                      [0.696, 0.391]

#The following variables are not required but recommended.
#You can leave these empty if you wish.

SoftwareVersion
ScreenAOIDefinition
EyeCameraSettings
EyeTrackerDistance              60
FeatureDetectionSettings
GazeMappingSettings
RawDataFilters
InstitutionName	                Best Lab Ever
InstitutionAddress
TaskName	                    awesome-taskname
```


## Run code

```bash
eye2bids [-h] [-v] --input_file INPUT_FILE [--metadata_file METADATA_FILE] [--output_dir OUTPUT_DIR] [-i] [--verbosity {0,1,2,3}] [--input_type INPUT_TYPE]
```

minimum:
```bash
eye2bids --input_file INPUT_FILE
```

```
Options:
-h, --help                      show this help message and exit
-v, --version                   show program's version number and exit
--input_file INPUT_FILE         Path to the input file to convert.
--metadata_file METADATA_FILE   Path to the a tsv file containing extra metadata.
--output_dir OUTPUT_DIR         path to output directory.
-i, --interactive               To run in interactive mode.
--verbosity {0,1,2,3}           Verbosity level.
-f, --force                     To run the converter without passing a metadata.yml file. Creates an invalid BIDS dataset.
--input_type INPUT_TYPE         Input type if it cannot be determined from input_file.
```

## Output

If everything went well, you should find the following data in your `output_dir`:

```
├─ sub-01
│  └─ beh
│      ├─ sub-01_task-FreeView_recording-eye1_physio.json
│      ├─ sub-01_task-FreeView_recording-eye1_physio.tsv.gz
│      ├─ sub-01_task-FreeView_recording-eye1_physioevents.json
│      ├─ sub-01_task-FreeView_recording-eye1_physioevents.tsv.gz
│      ├─ sub-01_task-FreeView_events.json
│      └─ sub-01_task-FreeView_events.tsv
```

!!! info "Other files required by BIDS"

    Please note that this software does not create other files that are required for a valid BIDS dataset, such as `dataset_description.json` or a `participants.tsv` table etc. For this, there are already other software available which you can find [here]().


## Docker

You can build the docker image with the following command:

```bash
docker build -t eye2bids:latest .
```



[^1]: Values and units are examples and need to be adjusted by you. Please visit the [specification]() to check other accepted units.
