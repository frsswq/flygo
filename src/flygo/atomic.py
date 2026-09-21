"""Atomic replacement of generated artifacts.

Every writer in this repository publishes a file by writing a sibling temporary
file and replacing the target, so an interrupted run cannot leave a truncated
artifact where a reader expects a complete one.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, TextIO


def _write(path: Path, mode: str, payload: Callable[..., object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, mode) as temporary:
            payload(temporary)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_bytes(path: Path, payload: Callable[[BinaryIO], object]) -> None:
    """Replace ``path`` with the bytes that ``payload`` writes."""
    _write(path, "wb", payload)


def write_text(path: Path, payload: Callable[[TextIO], object]) -> None:
    """Replace ``path`` with the text that ``payload`` writes."""
    _write(path, "w", payload)
