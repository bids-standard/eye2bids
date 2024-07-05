from argparse import ArgumentParser
from pathlib import Path

from rich_argparse import RichHelpFormatter

from eye2bids._version import __version__


def global_parser() -> ArgumentParser:
    """Parse command line arguments.

    Returns
    -------
        parser: An ArgumentParser object.

    """
    parser = ArgumentParser(
        prog="eye2bids",
        description="Converts eyetracking data to a BIDS compatible format.",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"{__version__}",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="""
        Path to the input file to convert.
        """,
        required=True,
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        help="""
        Path to the a yaml file containing extra metadata.
        """,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="""
        Path to output directory.
        """,
        required=False,
        default=Path.cwd(),
    )
    parser.add_argument(
        "-i",
        "--interactive",
        help="""
        To run in interactive mode.
        """,
        action="store_true",
    )
    parser.add_argument(
        "--verbosity",
        help="""
        Verbosity level.
        """,
        required=False,
        choices=[0, 1, 2, 3],
        default=2,
        type=int,
        nargs=2,
    )
    parser.add_argument(
        "-f",
        "--force",
        help="""
        To run the converter without passing a metadata.yml file.\n 
        Creates an invalid BIDS dataset.
        """,
        action="store_true",
    )
    return parser
