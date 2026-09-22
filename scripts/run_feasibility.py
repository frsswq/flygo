"""Validate or dry-run the bounded 19x19 feasibility experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from flygo.feasibility import FeasibilityProtocol, dry_run, validate_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/protocols/feasibility-19-v1.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    try:
        protocol = FeasibilityProtocol.model_validate_json(arguments.protocol.read_bytes())
        result = (
            dry_run(protocol)
            if arguments.dry_run
            else {
                "protocol": protocol.name,
                "blockers": validate_protocol(protocol),
            }
        )
    except (OSError, ValueError, ValidationError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
