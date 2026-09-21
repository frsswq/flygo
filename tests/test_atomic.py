from pathlib import Path

import pytest

from flygo.atomic import write_bytes, write_text


def test_write_bytes_replaces_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"old")

    write_bytes(path, lambda stream: stream.write(b"new"))

    assert path.read_bytes() == b"new"


def test_write_text_creates_missing_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.json"

    write_text(path, lambda stream: stream.write("{}\n"))

    assert path.read_text() == "{}\n"


def test_failed_write_keeps_the_previous_file_and_no_temporary(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"old")

    def failing(stream: object) -> None:
        del stream
        raise RuntimeError("injected write failure")

    with pytest.raises(RuntimeError, match="injected write failure"):
        write_bytes(path, failing)

    assert path.read_bytes() == b"old"
    assert [entry.name for entry in tmp_path.iterdir()] == ["artifact.bin"]


def test_failed_first_write_publishes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"

    def failing(stream: object) -> None:
        del stream
        raise RuntimeError("injected write failure")

    with pytest.raises(RuntimeError, match="injected write failure"):
        write_text(path, failing)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
