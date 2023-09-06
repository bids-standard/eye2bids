# eye2bids

## Instructions for testing conversion_json

### Requirements

- Python3

- EyeLink Developers Kit. Download from [SR-Research support forum] (forum registration required)

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
