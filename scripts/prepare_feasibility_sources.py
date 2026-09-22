"""Pin enough official KataGo game archives for the feasibility corpus."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from pydantic import ValidationError

from flygo.atomic import write_text
from flygo.feasibility import FeasibilitySources, file_sha256

PAGES = (
    "https://katagotraining.org/networks/kata1/kata1-b18c384nbt-s6476684544-d3393539279/training-games/",
    "https://katagotraining.org/networks/kata1/kata1-b18c384nbt-s6706197504-d3452137032/training-games/",
    "https://katagotraining.org/networks/kata1/kata1-b18c384nbt-s6890580736-d3500282628/training-games/",
    "https://katagotraining.org/networks/kata1/kata1-b18c384nbt-s6981484800-d3524616345/training-games/",
    "https://katagotraining.org/networks/kata1/kata1-b18c384nbt-s7131841024-d3563447968/training-games/",
    "https://katagotraining.org/networks/kata1/kata1-b18c384nbt-s7161915648-d3571334277/training-games/",
    "https://katagotraining.org/networks/kata1/kata1-b18c384nbt-s7560105728-d3675868361/training-games/",
    "https://katagotraining.org/networks/kata1/kata1-b18c384nbt-s7860356864-d3754557269/training-games/",
    "https://katagotraining.org/networks/kata1/kata1-b18c384nbt-s8160926464-d3833541428/training-games/",
    "https://katagotraining.org/networks/kata1/kata1-b18c384nbt-s8768585216-d3995752096/training-games/",
)


def download(url: str, path: Path) -> None:
    if path.exists():
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
                url,
                "--output",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_existing(manifest: FeasibilitySources, root: Path) -> None:
    for page in manifest.pages:
        network = page.url.rstrip("/").split("/")[-2]
        path = root / "training-candidates" / f"{network}.html"
        download(page.url, path)
        if file_sha256(path) != page.sha256:
            raise ValueError(f"Source page SHA256 mismatch: {path}")
    for archive in manifest.archives:
        path = root / archive.file
        download(archive.url, path)
        if file_sha256(path) != archive.sha256:
            raise ValueError(f"Archive SHA256 mismatch: {path}")


def prepare(root: Path, output: Path) -> FeasibilitySources:
    if output.exists():
        manifest = FeasibilitySources.model_validate_json(output.read_bytes())
        verify_existing(manifest, root)
        return manifest

    page_records = []
    archive_urls: list[str] = []
    for url in PAGES:
        network = url.rstrip("/").split("/")[-2]
        page_path = root / "training-candidates" / f"{network}.html"
        download(url, page_path)
        game_ids = re.findall(r'href="/sgfplayer/training-games/(\d+)/"', page_path.read_text())[
            ::10
        ]
        if not game_ids:
            raise ValueError(f"No training games found on {url}")
        page_records.append(
            {
                "url": url,
                "sha256": file_sha256(page_path),
                "sampled_game_ids": game_ids,
            }
        )
        for game_id in game_ids:
            metadata_path = root / "training-candidates" / f"{game_id}.json"
            download(
                f"https://katagotraining.org/api/games/training/{game_id}/",
                metadata_path,
            )
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid game metadata: {metadata_path}") from error
            archive_urls.append(str(metadata["sgf_file"]))

    archives = []
    for url in dict.fromkeys(archive_urls):
        name = Path(urlparse(url).path).name
        if not name or name == ".":
            raise ValueError(f"Archive URL has no filename: {url}")
        relative = f"archives/{name}"
        path = root / relative
        download(url, path)
        archives.append({"url": url, "file": relative, "sha256": file_sha256(path)})

    manifest = FeasibilitySources.model_validate(
        {
            "schema_version": 1,
            "selection": (
                "Ten fixed b18 training releases in listed order; every tenth game link "
                "from each captured training-games page, in page order"
            ),
            "pages": page_records,
            "archives": archives,
        }
    )
    write_text(output, lambda stream: stream.write(manifest.model_dump_json(indent=2) + "\n"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/teacher-pilot"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/protocols/feasibility-sources-v1.json"),
    )
    arguments = parser.parse_args()
    try:
        manifest = prepare(arguments.root, arguments.output)
    except (
        KeyError,
        OSError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        raise SystemExit(str(error)) from error
    print(f"{arguments.output}: {len(manifest.archives)} pinned archives")


if __name__ == "__main__":
    main()
