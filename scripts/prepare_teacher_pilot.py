"""Restore the pinned pilot inputs without running KataGo or opening test targets."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, field_validator

from flygo.dataset import load_sgf_games, parse_sgf_collection, split_of
from flygo.teacher import katago_queries

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class Download(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: HttpUrl
    file: str
    sha256: Sha256

    @field_validator("file")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("Expected a relative file path inside the pilot directory")
        return value


class GameSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    archive: str
    member: str
    sha256: Sha256
    game_id: str
    split: Literal["train", "validation", "test"]
    moves: int = Field(gt=0)


class Sources(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    selection: str
    configuration_sha256: Sha256
    visits: int = Field(gt=0)
    stride: int = Field(gt=0)
    downloads: list[Download] = Field(min_length=1)
    games: list[GameSource] = Field(min_length=1)

    @field_validator("stride")
    @classmethod
    def odd_stride(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("Pilot stride must be odd to sample both players")
        return value


def verify(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def download(asset: Download, root: Path) -> None:
    path = root / asset.file
    if path.exists():
        verify(path, asset.sha256)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "2",
                "--max-time",
                "300",
                str(asset.url),
                "--output",
                str(temporary),
            ],
            check=True,
        )
        verify(temporary, asset.sha256)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def publish(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(sources_path: Path, root: Path) -> None:
    sources = Sources.model_validate_json(sources_path.read_bytes())
    configuration = sources_path.parent / "analysis.cfg"
    verify(configuration, sources.configuration_sha256)
    assets = {asset.file: asset for asset in sources.downloads}
    if len(assets) != len(sources.downloads):
        raise ValueError("Duplicate download paths")
    if len({game.game_id for game in sources.games}) != len(sources.games):
        raise ValueError("Duplicate games")
    for game in sources.games:
        if f"archives/{game.archive}" not in assets:
            raise ValueError(f"Game references an unpinned archive: {game.archive}")
    for asset in sources.downloads:
        download(asset, root)
    if (root / "analysis.cfg").exists():
        verify(root / "analysis.cfg", sources.configuration_sha256)
    else:
        publish(root / "analysis.cfg", configuration.read_bytes())
    sgf_directory = root / "sgf"
    sgf_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for source in sources.games:
        # Read one member, never extract archive paths onto the filesystem.
        with tarfile.open(root / "archives" / source.archive) as archive:
            member = archive.getmember(source.member)
            if not member.isfile() or member.size > 1_000_000:
                raise ValueError(f"Invalid SGF member: {source.member}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing SGF member: {source.member}")
            with stream:
                payload = stream.read()
        if hashlib.sha256(payload).hexdigest() != source.sha256:
            raise ValueError(f"SGF SHA256 mismatch: {source.member}")
        games = parse_sgf_collection(payload)
        if len(games) != 1:
            raise ValueError("Expected one game per pilot SGF")
        (game,) = games
        if (game.game_id, game.size, len(game.moves), split_of(game.game_id)) != (
            source.game_id,
            19,
            source.moves,
            source.split,
        ):
            raise ValueError(f"SGF metadata mismatch: {source.member}")
        path = sgf_directory / PurePosixPath(source.member).name
        if path.exists():
            verify(path, source.sha256)
        else:
            publish(path, payload)
        paths.append(path)
    if set(sgf_directory.glob("*.sgf")) != set(paths):
        raise ValueError("Unexpected SGF files in the pilot directory; use a clean directory")
    queries = list(
        katago_queries(load_sgf_games(paths), visits=sources.visits, stride=sources.stride)
    )
    content = "".join(json.dumps(query) + "\n" for query in queries)
    query_path = root / "queries.jsonl"
    if query_path.exists():
        if query_path.read_text() != content:
            raise ValueError("Existing queries differ; use a clean directory")
    else:
        publish(query_path, content.encode())
    positions = sum(len(query["analyzeTurns"]) for query in queries)
    print(f"{root}: {len(queries)} games, {positions} requested positions; hashes verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("docs/teacher-pilot/sources.json"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/teacher-pilot"))
    arguments = parser.parse_args()
    try:
        arguments.root.mkdir(parents=True, exist_ok=True)
        with (arguments.root / ".prepare.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare(arguments.sources, arguments.root)
    except (
        ValueError,
        OSError,
        KeyError,
        tarfile.TarError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
