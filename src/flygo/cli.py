"""Command-line workflows for reproducible MaleCNS data preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flygo.official_data import download_official_files, prepare_traced_graph, write_profile

DEFAULT_RAW = Path("data/raw")
DEFAULT_GRAPH = Path("data/processed/malecns-traced-w5.parquet")
DEFAULT_PROFILE = Path("docs/data-profile.json")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="flygo")
    commands = root.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download and verify official MaleCNS files")
    download.add_argument("--destination", type=Path, default=DEFAULT_RAW)

    profile = commands.add_parser("profile", help="Measure schemas and graph counts")
    profile.add_argument("--data", type=Path, default=DEFAULT_RAW)
    profile.add_argument("--output", type=Path, default=DEFAULT_PROFILE)

    prepare = commands.add_parser("prepare", help="Build the traced sparse experiment graph")
    prepare.add_argument("--data", type=Path, default=DEFAULT_RAW)
    prepare.add_argument("--output", type=Path, default=DEFAULT_GRAPH)
    prepare.add_argument("--minimum-weight", type=int, default=5)
    return root


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "download":
        paths = download_official_files(arguments.destination)
        for path in paths:
            print(path)
    elif arguments.command == "profile":
        result = write_profile(arguments.data, arguments.output)
        print(json.dumps(result, indent=2))
    elif arguments.command == "prepare":
        if arguments.minimum_weight < 1:
            raise SystemExit("--minimum-weight must be at least 1")
        prepare_traced_graph(
            arguments.data,
            arguments.output,
            minimum_weight=arguments.minimum_weight,
        )
        print(arguments.output)


if __name__ == "__main__":
    main()
