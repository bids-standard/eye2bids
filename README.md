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

To try it, you can install our test data from OSF by running the following command:

```bash
python tools/download_test_data.py
```

- manual_metadata.yml file (find template and an example in conversion_json folder)

### Run code

```bash
python edf2bids_json.py
```

[SR-Research support forum]: https://www.sr-research.com/support/forum-9.html

## Docker

You can build the docker image with the following command:

```bash
docker build -t eye2bids:latest .
```

## Contributing

Make sure you install eye2bids in editable mode (see above) and install the development dependencies:

```bash
pip install --editable .[dev]
```

## Related projects
