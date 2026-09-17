"""Acquisition and profiling for the official Janelia MaleCNS v1.0 release."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.feather as feather

DATASET = "male-cns:v1.0"
BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"


@dataclass(frozen=True)
class OfficialFile:
    name: str
    sha256: str
    size_bytes: int

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.name}"


OFFICIAL_FILES = (
    OfficialFile(
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
        1_051_241_946,
    ),
    OfficialFile(
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
        14_483_314,
    ),
    OfficialFile(
        "body-neurotransmitters-male-cns-v1.0.feather",
        "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
        43_282_834,
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_official_files(destination: Path) -> list[Path]:
    """Download and verify the three official files used by FlyGo."""
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for item in OFFICIAL_FILES:
        target = destination / item.name
        if not target.exists() or file_sha256(target) != item.sha256:
            temporary = target.with_suffix(f"{target.suffix}.part")
            urllib.request.urlretrieve(item.url, temporary)
            temporary.replace(target)
        if target.stat().st_size != item.size_bytes or file_sha256(target) != item.sha256:
            raise ValueError(f"Checksum verification failed for {target}")
        downloaded.append(target)
    return downloaded


def read_annotations(path: Path) -> pl.DataFrame:
    """Read official annotations into Polars despite a nullable Arrow dictionary."""
    table = feather.read_table(path)
    columns: list[pa.ChunkedArray] = []
    for field, column in zip(table.schema, table.columns, strict=True):
        columns.append(column.cast(pa.string()) if pa.types.is_dictionary(field.type) else column)
    frame = pl.from_arrow(pa.table(columns, names=table.column_names))
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("Expected annotation table to produce a Polars DataFrame")
    return frame


def read_neurotransmitters(path: Path) -> pl.DataFrame:
    return pl.read_ipc(path)


def scan_connections(path: Path) -> pl.LazyFrame:
    """Scan the official segment graph and normalize its three column names."""
    return pl.scan_ipc(path).select(
        pl.col("body_pre").alias("pre"),
        pl.col("body_post").alias("post"),
        pl.col("weight"),
    )


def build_profile(data_directory: Path) -> dict[str, object]:
    """Measure the release rather than relying on paper-level headline counts."""
    paths = {item.name: data_directory / item.name for item in OFFICIAL_FILES}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing official files: {', '.join(missing)}")

    annotations = read_annotations(paths[OFFICIAL_FILES[1].name])
    neurotransmitters = read_neurotransmitters(paths[OFFICIAL_FILES[2].name])
    traced_ids = annotations.filter(pl.col("status") == "Traced")["bodyId"]
    connections = scan_connections(paths[OFFICIAL_FILES[0].name])
    traced = connections.filter(
        pl.col("pre").is_in(traced_ids.implode()) & pl.col("post").is_in(traced_ids.implode())
    )
    thresholds = (1, 2, 5, 10, 20, 50)
    graph_summary = (
        traced.select(
            pl.len().alias("edges"),
            pl.col("weight").sum().alias("synaptic_contacts"),
            *[
                (pl.col("weight") >= value).sum().alias(f"edges_weight_ge_{value}")
                for value in thresholds
            ],
        )
        .collect()
        .row(0, named=True)
    )

    return {
        "dataset": DATASET,
        "source": "https://male-cns.janelia.org/download/",
        "files": [asdict(item) | {"url": item.url} for item in OFFICIAL_FILES],
        "annotations": {
            "rows": annotations.height,
            "traced_neurons": traced_ids.len(),
            "columns": annotations.columns,
        },
        "neurotransmitters": {
            "rows": neurotransmitters.height,
            "columns": neurotransmitters.columns,
        },
        "traced_graph": graph_summary,
    }


def write_profile(data_directory: Path, destination: Path) -> dict[str, object]:
    profile = build_profile(data_directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(profile, indent=2) + "\n")
    return profile


def prepare_traced_graph(
    data_directory: Path,
    destination: Path,
    *,
    minimum_weight: int = 5,
) -> None:
    """Write the experiment graph with both endpoints restricted to traced neurons."""
    annotations = read_annotations(data_directory / OFFICIAL_FILES[1].name)
    traced_ids = annotations.filter(pl.col("status") == "Traced")["bodyId"]
    graph = scan_connections(data_directory / OFFICIAL_FILES[0].name).filter(
        (pl.col("weight") >= minimum_weight)
        & pl.col("pre").is_in(traced_ids.implode())
        & pl.col("post").is_in(traced_ids.implode())
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    graph.sink_parquet(destination)
