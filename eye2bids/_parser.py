import argparse


def parser():
    """Parse command line arguments.

    Returns
    -------
        parser: An ArgumentParser object.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--metadata_file", type=str)
    parser.add_argument("--output_dir", type=str)
