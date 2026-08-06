"""Fast, dependency-free unit tests for M9-16: feature_pipeline_version() is
a sha256 content hash of dataset_builder.py + train_pipeline.py +
config.yaml, not a single-file git hash. No DB, no MLflow, no `.git/`
required -- the function only reads local file bytes, so these tests point
`_PIPELINE_FILES` at throwaway tmp_path files rather than touching the real
repo files.
"""

from __future__ import annotations

from pathlib import Path

import domains.pharma.dataset_builder as dataset_builder


def _write(path: Path, content: str) -> None:
    path.write_text(content)


def _three_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    a = tmp_path / "dataset_builder.py"
    b = tmp_path / "train_pipeline.py"
    c = tmp_path / "config.yaml"
    _write(a, "a=1")
    _write(b, "b=1")
    _write(c, "c: 1")
    return a, b, c


def test_hash_changes_when_dataset_builder_changes(tmp_path, monkeypatch):
    a, b, c = _three_files(tmp_path)
    monkeypatch.setattr(dataset_builder, "_PIPELINE_FILES", (a, b, c))
    baseline = dataset_builder.feature_pipeline_version()

    _write(a, "a=2")
    assert dataset_builder.feature_pipeline_version() != baseline


def test_hash_changes_when_train_pipeline_changes(tmp_path, monkeypatch):
    a, b, c = _three_files(tmp_path)
    monkeypatch.setattr(dataset_builder, "_PIPELINE_FILES", (a, b, c))
    baseline = dataset_builder.feature_pipeline_version()

    _write(b, "b=2")
    assert dataset_builder.feature_pipeline_version() != baseline


def test_hash_changes_when_config_yaml_changes(tmp_path, monkeypatch):
    a, b, c = _three_files(tmp_path)
    monkeypatch.setattr(dataset_builder, "_PIPELINE_FILES", (a, b, c))
    baseline = dataset_builder.feature_pipeline_version()

    _write(c, "c: 2")
    assert dataset_builder.feature_pipeline_version() != baseline


def test_hash_stable_when_unrelated_file_changes(tmp_path, monkeypatch):
    a, b, c = _three_files(tmp_path)
    unrelated = tmp_path / "unrelated.py"
    _write(unrelated, "x=1")
    monkeypatch.setattr(dataset_builder, "_PIPELINE_FILES", (a, b, c))
    baseline = dataset_builder.feature_pipeline_version()

    _write(unrelated, "x=999999")
    assert dataset_builder.feature_pipeline_version() == baseline


def test_hash_deterministic_for_identical_content(tmp_path, monkeypatch):
    a, b, c = _three_files(tmp_path)
    monkeypatch.setattr(dataset_builder, "_PIPELINE_FILES", (a, b, c))

    assert dataset_builder.feature_pipeline_version() == dataset_builder.feature_pipeline_version()
