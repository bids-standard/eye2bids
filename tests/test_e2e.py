from pathlib import Path

from eye2bids.edf2bids import main

# import pytest


# @pytest.mark.skip(reason="Not implemented yet")
def test_edf_end_to_end():
    data_dir = Path(__file__).parent / "data"
    input_file = data_dir / "decisions_modality_baptisteC2_7-5-2016_21-26-13.edf"
    metadata_file = data_dir / "metadata.yml"
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)

    main(
        input_file=input_file,
        metadata_file=metadata_file,
        output_dir=output_dir,
    )
