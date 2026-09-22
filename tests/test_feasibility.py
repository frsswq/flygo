import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from flygo.feasibility import FeasibilityProtocol, dry_run, validate_protocol
from flygo.research import MODELS


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def protocol_payload(tmp_path: Path) -> dict[str, object]:
    immutable = tmp_path / "input"
    immutable.write_bytes(b"fixed")
    pinned = artifact(immutable)
    return {
        "schema_version": 1,
        "name": "feasibility-19-v1",
        "source_revision": "34264a9",
        "claim_scope": "feasibility-only",
        "corpus": {
            "board_size": 19,
            "komi": 7.5,
            "target_labelled_positions": 10_000,
            "visits": 256,
            "stride": 1,
            "source_manifest": pinned,
            "data_permission": pinned,
            "teacher_network": pinned,
            "teacher_executable": pinned,
            "teacher_configuration": pinned,
            "output": {"path": str(tmp_path / "dataset.json"), "sha256": None},
            "timed_batch_positions": 100,
            "timed_batch_report": {"path": str(tmp_path / "timing.json"), "sha256": None},
        },
        "circuit": {
            "nodes": 500,
            "source_graph": pinned,
            "output": {"path": str(tmp_path / "circuit.parquet"), "sha256": None},
        },
        "training": {
            "models": list(MODELS),
            "seeds": [7, 17, 27],
            "epochs": 10,
            "batch_size": 128,
            "learning_rate": 0.001,
            "steps": 8,
            "retention": 0.35,
            "recurrent_gain": 0.9,
            "normalization": "incoming",
            "weights": "weighted",
            "value_weight": 1.0,
            "final_test": False,
            "output": str(tmp_path / "runs"),
            "summary": str(tmp_path / "runs" / "summary.json"),
        },
        "approvals": {
            "timed_batch_approved": True,
            "full_labelling_approved": False,
        },
    }


def test_feasibility_dry_run_lists_every_cell_without_opening_test_data(tmp_path: Path) -> None:
    protocol = FeasibilityProtocol.model_validate(protocol_payload(tmp_path))

    result = dry_run(protocol)

    assert result["status"] == "blocked"
    assert result["training"]["cell_count"] == 18
    assert result["training"]["final_test"] is False
    assert result["teacher"]["target_labelled_positions"] == 10_000
    assert any("awaits the timed-batch estimate" in item for item in result["blockers"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stride", 2, "stride must be odd"),
        ("seeds", [7, 7, 27], "seeds must be exactly"),
        ("models", list(reversed(MODELS)), "models must be exactly"),
    ],
)
def test_feasibility_rejects_changed_fixed_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = protocol_payload(tmp_path)
    section = payload["corpus"] if field == "stride" else payload["training"]
    section[field] = value  # type: ignore[index]
    protocol = FeasibilityProtocol.model_validate(payload)

    with pytest.raises(ValueError, match=message):
        validate_protocol(protocol)


def test_committed_feasibility_protocol_has_a_non_executing_dry_run() -> None:
    run = subprocess.run(
        [
            sys.executable,
            "scripts/run_feasibility.py",
            "--protocol",
            "docs/protocols/feasibility-19-v1.json",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["status"] == "blocked"
    assert result["training"]["cell_count"] == 18
    assert result["training"]["final_test"] is False
