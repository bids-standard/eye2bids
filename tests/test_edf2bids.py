import json
from pathlib import Path

import pytest

from eye2bids.edf2bids import _check_edf2asc_present, _convert_edf_to_asc, edf2bids


def data_dir():
    return Path(__file__).parent / "data"


def edf_test_files(input_dir=data_dir()):
    files = list(input_dir.glob("**/*.edf"))
    EDF_files = list(input_dir.glob("**/*.EDF"))
    files.extend(EDF_files)
    return files


@pytest.mark.parametrize("input_file", edf_test_files())
def test_convert_edf_to_asc(input_file):
    if not _check_edf2asc_present():
        pytest.skip("edf2asc missing")
    asc_file = _convert_edf_to_asc(input_file)
    assert Path(asc_file).exists()


@pytest.mark.parametrize("metadata_file", [data_dir() / "metadata.yml", None])
def test_edf_end_to_end(metadata_file):
    if not _check_edf2asc_present():
        pytest.skip("edf2asc missing")

    input_file = edf_test_files(input_dir=data_dir() / "osf" / "eyelink" / "decisions")[0]

    output_dir = data_dir() / "output"
    output_dir.mkdir(exist_ok=True)

    edf2bids(
        input_file=input_file,
        metadata_file=metadata_file,
        output_dir=output_dir,
    )

    assert (output_dir / "events.json").exists()
    with open(output_dir / "events.json") as f:
        events = json.load(f)
    assert events["StimulusPresentation"]["ScreenResolution"] == [1919, 1079]

    assert (output_dir / "eyetrack.json").exists()
    with open(output_dir / "eyetrack.json") as f:
        eyetrack = json.load(f)
    assert eyetrack["SamplingFrequency"] == 1000
    assert eyetrack["RecordedEye"] == "Right"
