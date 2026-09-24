"""Validate a completed matched research run and write its summary.

The summary is derived from the run directory, never hand-written.
It refuses an incomplete grid, a non-finite metric, a checkpoint that fails its hash,
a run that read the closed test split, a request that does not match the protocol,
and a control that cannot support attribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from flygo.atomic import write_text
from flygo.feasibility import FeasibilityProtocol
from flygo.official_data import file_sha256
from flygo.research import ResearchConfig
from flygo.summary import render_markdown, summarize


def research_config(protocol: FeasibilityProtocol) -> ResearchConfig:
    """Build the expected runner configuration from a feasibility protocol."""
    return ResearchConfig(
        size=protocol.corpus.board_size,
        seeds=protocol.training.seeds,
        epochs=protocol.training.epochs,
        batch_size=protocol.training.batch_size,
        learning_rate=protocol.training.learning_rate,
        steps=protocol.training.steps,
        retention=protocol.training.retention,
        recurrent_gain=protocol.training.recurrent_gain,
        normalization=protocol.training.normalization,
        weights=protocol.training.weights,
        value_weight=protocol.training.value_weight,
        final_test=protocol.training.final_test,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/protocols/feasibility-19-v1.json"),
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    try:
        protocol = FeasibilityProtocol.model_validate_json(arguments.protocol.read_bytes())
        dataset_path = protocol.corpus.output.path
        summary = summarize(
            runs=arguments.runs,
            config=research_config(protocol),
            models=protocol.training.models,
            dataset_manifest_sha256=(file_sha256(dataset_path) if dataset_path.is_file() else None),
            graph_file_sha256=file_sha256(protocol.circuit.output.path),
        )
        output = arguments.output if arguments.output is not None else protocol.training.summary
        markdown_path = output.with_suffix(".md")
        write_text(output, lambda stream: stream.write(json.dumps(summary, indent=2) + "\n"))
        write_text(markdown_path, lambda stream: stream.write(render_markdown(summary)))
    except (OSError, ValueError, ValidationError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "summary": str(output),
                "markdown": str(markdown_path),
                "verdict": summary["verdict"],
                "ranking": summary["ranking"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
