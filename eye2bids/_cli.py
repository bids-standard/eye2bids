from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from eye2bids._parser import global_parser
from eye2bids.edf2bids import edf2bids
from eye2bids.logger import eye2bids_logger

e2b_log = eye2bids_logger()


def set_verbosity(verbosity: int | list[int]) -> None:
    """Set verbosity level."""
    if isinstance(verbosity, list):
        verbosity = verbosity[0]
    verbosity_map = {0: "ERROR", 1: "WARNING", 2: "INFO", 3: "DEBUG"}
    e2b_log.setLevel(verbosity_map[verbosity])


def cli(argv: Sequence[str] = sys.argv) -> None:
    """Entry point."""
    parser = global_parser()
    parser.add_argument(
        "--input_type",
        type=str,
        help="""
        Input type if it cannot be determined from input_file.
        """,
        required=False,
        default=None,
    )

    args, _ = parser.parse_known_args(argv[1:])

    input_file = Path(args.input_file).absolute()

    metadata_file = args.metadata_file
    if metadata_file not in [None, ""]:
        metadata_file = Path(metadata_file).absolute()

    output_dir = Path(args.output_dir).resolve()

    set_verbosity(args.verbosity)

    edf2bids(
        input_file=input_file,
        metadata_file=metadata_file,
        output_dir=output_dir,
        interactive=args.interactive,
        force=args.force,
    )
