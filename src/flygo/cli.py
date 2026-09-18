"""Command-line workflows for reproducible MaleCNS data preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flygo.conformance import DEFAULT_CONFORMANCE, write_conformance
from flygo.export import DEFAULT_BUNDLE, estimate_bundle, write_web_bundle
from flygo.official_data import download_official_files, prepare_traced_graph, write_profile
from flygo.policy_fixture import DEFAULT_POLICY_CONFORMANCE, write_policy_conformance

DEFAULT_RAW = Path("data/raw")
DEFAULT_GRAPH = Path("data/processed/malecns-traced-w5.parquet")
DEFAULT_PROFILE = Path("docs/data-profile.json")


def _sgf_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(item.rglob("*.sgf"))
        elif item.suffix.lower() == ".sgf":
            paths.append(item)
        else:
            raise SystemExit(f"Not an SGF file or directory: {item}")
    if not paths:
        raise SystemExit("No SGF files found")
    return sorted(set(paths))


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

    dataset = commands.add_parser("build-dataset", help="Build split policy-value data from SGF")
    dataset.add_argument("--sgf", type=Path, nargs="+", required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--size", type=int, default=19)
    dataset.add_argument("--stride", type=int, default=1)
    dataset.add_argument("--teacher", type=Path)

    queries = commands.add_parser("teacher-queries", help="Write KataGo analysis JSON Lines")
    queries.add_argument("--sgf", type=Path, nargs="+", required=True)
    queries.add_argument("--output", type=Path, required=True)
    queries.add_argument("--visits", type=int, default=256)

    teacher = commands.add_parser("teacher-import", help="Import KataGo analysis JSON Lines")
    teacher.add_argument("--sgf", type=Path, nargs="+", required=True)
    teacher.add_argument("--analysis", type=Path, required=True)
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument(
        "--winrate-perspective",
        choices=("black", "white", "side-to-move"),
        default="black",
        help="The reportAnalysisWinratesAs value used by KataGo",
    )

    conformance = commands.add_parser(
        "conformance",
        help="Regenerate the shared rules conformance fixture",
    )
    conformance.add_argument("--output", type=Path, default=DEFAULT_CONFORMANCE)

    export = commands.add_parser("export-web", help="Write the browser graph and policy bundle")
    export.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)

    policy_fixture = commands.add_parser(
        "policy-conformance",
        help="Regenerate the shared browser policy fixture",
    )
    policy_fixture.add_argument("--output", type=Path, default=DEFAULT_POLICY_CONFORMANCE)
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
    elif arguments.command == "build-dataset":
        from flygo.dataset import build_dataset, load_sgf_games

        games = load_sgf_games(_sgf_paths(arguments.sgf))
        result = build_dataset(
            games,
            arguments.output,
            size=arguments.size,
            stride=arguments.stride,
            teacher_path=arguments.teacher,
        )
        print(
            f"{arguments.output}: {result.accepted_games} games, "
            f"{result.examples} examples, {result.rejected_games} rejected"
        )
    elif arguments.command == "teacher-queries":
        from flygo.dataset import load_sgf_games
        from flygo.teacher import write_katago_queries

        games = load_sgf_games(_sgf_paths(arguments.sgf))
        count = write_katago_queries(games, arguments.output, visits=arguments.visits)
        print(f"{arguments.output}: {count} queries")
    elif arguments.command == "teacher-import":
        from flygo.dataset import load_sgf_games
        from flygo.teacher import import_katago_analysis

        games = load_sgf_games(_sgf_paths(arguments.sgf))
        count = import_katago_analysis(
            arguments.analysis,
            arguments.output,
            games,
            winrate_perspective=arguments.winrate_perspective,
        )
        print(f"{arguments.output}: {count} targets")
    elif arguments.command == "conformance":
        fixture = write_conformance(arguments.output)
        print(
            f"{arguments.output}: {len(fixture['cases'])} cases, {len(fixture['illegal'])} illegal"
        )
    elif arguments.command == "export-web":
        manifest = write_web_bundle(arguments.output)
        sizes = estimate_bundle(arguments.output)
        print(
            f"{arguments.output}: {manifest['graph']['node_count']} neurons, {sizes['total']} bytes"
        )
    elif arguments.command == "policy-conformance":
        fixture = write_policy_conformance(arguments.output)
        print(f"{arguments.output}: {len(fixture['cases'])} cases")


if __name__ == "__main__":
    main()
