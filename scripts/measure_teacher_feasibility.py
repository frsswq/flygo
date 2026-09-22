"""Measure the pinned KataGo teacher on a bounded feasibility batch."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import resource
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from flygo.atomic import write_bytes, write_text
from flygo.feasibility import (
    FeasibilityProtocol,
    file_sha256,
    timing_queries,
    validate_protocol,
    verify_timing_analysis,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def _windows_path(path: Path) -> str:
    result = subprocess.run(
        ["wslpath", "-w", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def engine_command(protocol: FeasibilityProtocol) -> list[str]:
    engine = protocol.corpus.teacher_executable.path
    configuration = protocol.corpus.teacher_configuration.path
    network = protocol.corpus.teacher_network.path
    if engine.suffix.lower() == ".exe":
        configuration_argument = _windows_path(configuration)
        network_argument = _windows_path(network)
    else:
        configuration_argument = str(configuration)
        network_argument = str(network)
    return [
        str(engine),
        "analysis",
        "-config",
        configuration_argument,
        "-model",
        network_argument,
    ]


def _reuse_report(protocol: FeasibilityProtocol) -> dict[str, Any] | None:
    report_path = protocol.corpus.timed_batch_report.path
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Existing timing report is invalid: {report_path}") from error
    if report.get("schema_version") != 1 or report.get("protocol") != protocol.name:
        raise ValueError(f"Existing timing report has the wrong identity: {report_path}")
    for name in ("queries", "analysis", "stderr"):
        artifact = report["artifacts"][name]
        path = Path(artifact["path"])
        if file_sha256(path) != artifact["sha256"]:
            raise ValueError(f"Existing timing artifact is damaged: {path}")
    if protocol.corpus.timed_batch_report.sha256 is not None:
        actual = file_sha256(report_path)
        if actual != protocol.corpus.timed_batch_report.sha256:
            raise ValueError(f"Existing timing report SHA256 mismatch: {report_path}")
    return report


def measure(protocol: FeasibilityProtocol, *, timeout: float) -> dict[str, Any]:
    blockers = validate_protocol(protocol)
    disallowed = [
        blocker
        for blocker in blockers
        if not blocker.startswith(("timed batch report", "feasibility dataset", "selected circuit"))
        and "full 10,000-position labelling run awaits" not in blocker
    ]
    if disallowed:
        raise ValueError("Cannot run timed batch: " + "; ".join(disallowed))
    if not protocol.approvals.timed_batch_approved:
        raise ValueError("Timed teacher batch is not approved")

    reused = _reuse_report(protocol)
    if reused is not None:
        return reused

    report_path = protocol.corpus.timed_batch_report.path
    root = report_path.parent
    root.mkdir(parents=True, exist_ok=True)
    queries = timing_queries(
        protocol.corpus.pilot_queries.path,
        protocol.corpus.timed_batch_positions,
    )
    query_bytes = b"".join(json.dumps(query).encode() + b"\n" for query in queries)
    query_path = root / "queries.jsonl"
    analysis_path = root / "analysis.jsonl"
    stderr_path = root / "analysis.log"

    temporary_paths: list[Path] = []
    try:
        handles: list[tuple[int, str]] = [
            tempfile.mkstemp(dir=root, prefix=".analysis."),
            tempfile.mkstemp(dir=root, prefix=".analysis-log."),
        ]
        temporary_paths = [Path(name) for _, name in handles]
        analysis_descriptor, _ = handles[0]
        stderr_descriptor, _ = handles[1]
        command = engine_command(protocol)
        before_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        started = time.monotonic()
        with (
            os.fdopen(analysis_descriptor, "wb") as analysis_stream,
            os.fdopen(stderr_descriptor, "wb") as stderr_stream,
        ):
            result = subprocess.run(
                command,
                input=query_bytes,
                stdout=analysis_stream,
                stderr=stderr_stream,
                timeout=timeout,
            )
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise ValueError(f"Teacher exited with status {result.returncode}")
        completed = verify_timing_analysis(
            temporary_paths[0],
            queries,
            visits=protocol.corpus.visits,
        )
        after_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

        write_bytes(query_path, lambda stream: stream.write(query_bytes))
        temporary_paths[0].replace(analysis_path)
        temporary_paths[1].replace(stderr_path)
        positions_per_second = completed / elapsed
        target = protocol.corpus.target_labelled_positions
        report: dict[str, Any] = {
            "schema_version": 1,
            "protocol": protocol.name,
            "source_revision": protocol.source_revision,
            "positions": completed,
            "visits_per_position": protocol.corpus.visits,
            "elapsed_seconds": elapsed,
            "positions_per_second": positions_per_second,
            "projection": {
                "positions": target,
                "seconds": target / positions_per_second,
                "hours": target / positions_per_second / 3600,
                "analysis_bytes": round(analysis_path.stat().st_size * target / completed),
                "query_bytes": round(query_path.stat().st_size * target / completed),
            },
            "memory": {
                "method": "resource.getrusage(RUSAGE_CHILDREN).ru_maxrss",
                "scope": "maximum child-process RSS observed by WSL; Windows GPU memory excluded",
                "peak_rss_kib": max(before_rss, after_rss),
            },
            "artifacts": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in {
                    "queries": query_path,
                    "analysis": analysis_path,
                    "stderr": stderr_path,
                    "teacher_executable": protocol.corpus.teacher_executable.path,
                    "teacher_network": protocol.corpus.teacher_network.path,
                    "teacher_configuration": protocol.corpus.teacher_configuration.path,
                    "source_queries": protocol.corpus.pilot_queries.path,
                }.items()
            },
        }
        write_text(report_path, lambda stream: stream.write(_json_bytes(report).decode()))
        return report
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/protocols/feasibility-19-v1.json"),
    )
    parser.add_argument("--timeout", type=float, default=7200)
    arguments = parser.parse_args()
    try:
        protocol = FeasibilityProtocol.model_validate_json(arguments.protocol.read_bytes())
        report_path = protocol.corpus.timed_batch_report.path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with (report_path.parent / ".timing.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = measure(protocol, timeout=arguments.timeout)
    except (
        KeyError,
        OSError,
        ValueError,
        ValidationError,
        subprocess.SubprocessError,
    ) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
