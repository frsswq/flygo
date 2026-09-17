from flygo.official_data import DATASET, OFFICIAL_FILES


def test_official_manifest_is_pinned_and_uses_secure_checksums() -> None:
    assert DATASET == "male-cns:v1.0"
    assert len(OFFICIAL_FILES) == 3
    assert all(len(item.sha256) == 64 for item in OFFICIAL_FILES)
    assert all(
        item.url.startswith("https://storage.googleapis.com/flyem-male-cns/")
        for item in OFFICIAL_FILES
    )
