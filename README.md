# eye2bids

## Instructions for testing conversion_json

### Requirements

- Python >= 3.8

- EyeLink Developers Kit. Download from [SR-Research support forum] (forum registration required)

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

- Input data:

Those can be:

    - edf file by EyeLink Eye Tracker

You can install uur test data from OSF by running the following command:

```bash
python tools/download_test_data.py
```

- manual_metadata.yml file (find template and an example in conversion_json folder)

### Run code

```bash
python edf2bids_json.py
```

[SR-Research support forum]: https://www.sr-research.com/support/forum-9.html


## Related projects
