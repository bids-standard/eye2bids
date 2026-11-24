# How to contribute

## Install

- fork the repository
- clone your fork

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/eye2bids.git
```

- install the package in editable mode with all its development dependencies

```bash
cd eye2bids
pip install -e '.[dev]'
```

## Test


To run the tests, you need to install the [test data from OSF](https://osf.io/jdv7n/)
by running the following command:

```bash
python tools/download_test_data.py
```

You can then run any test by using pytest

```bash
python -m pytest tests/path_to_test_file.py::function_to_run
```

For example:
```bash
python -m pytest tests/test_edf2bids.py::test_convert_edf_to_asc_events
```
