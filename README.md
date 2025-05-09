[![test](https://github.com/bids-standard/eye2bids/actions/workflows/tests.yml/badge.svg)](https://github.com/bids-standard/eye2bids/actions/workflows/tests.yml)

# eye2bids

## Installation

### Requirements

- Python >= 3.8

If you want to use eye2bids to convert EyeLink data,
you will need to install the EyeLink Developers Kit.
It can be downloaded from SR-Research support forum (forum registration required).

The installation on Ubuntu can also be done with the following commands:

Taken from: https://www.sr-research.com/support/docs.php?topic=linuxsoftware

```bash
sudo add-apt-repository universe
sudo apt update
sudo apt install ca-certificates
sudo apt-key adv --fetch-keys https://apt.sr-research.com/SRResearch_key
sudo add-apt-repository 'deb [arch=amd64] https://apt.sr-research.com SRResearch main'
sudo apt update
sudo apt install eyelink-display-software
```

### Install eye2bids

- Clone the repository

```bash
git clone https://github.com/bids-standard/eye2bids.git
```

- Install the package in editatble mode

```bash
cd eye2bids
pip install .
```

## Using eye2bids

- Supporeted Input data:

    - edf file by EyeLink Eye Tracker

- manual_metadata.yml file (find template and an example in conversion_json folder)

### Run code

```bash
eye2bids --input_file INPUT_FILE --metadata_file METADATA_FILE
```

    Usage: eye2bids [-h] [-v] --input_file INPUT_FILE --metadata_file METADATA_FILE [--output_dir OUTPUT_DIR] [-i] [--verbosity {0,1,2,3}]
                    [--input_type INPUT_TYPE]

    Converts eyetracking data to a BIDS compatible format.

    Options:
    -h, --help            show this help message and exit
    -v, --version         show program's version number and exit
    --input_file INPUT_FILE
                            Path to the input file to convert.
    --metadata_file METADATA_FILE
                            Path to the a yaml file containing extra metadata.
    --output_dir OUTPUT_DIR
                            Path to output directory.
    -i, --interactive     To run in interactive mode.
    --verbosity {0,1,2,3}
                            Verbosity level.
    --input_type INPUT_TYPE
                            Input type if it cannot be determined from input_file.

## Docker

You can build the docker image with the following command:

```bash
docker build -t eye2bids:latest .
```

## Related projects
