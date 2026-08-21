"""
tests/test_recorder.py
----------------------
Unit tests for the gesture data recorder.
"""

import json
import numpy as np
import pytest
from pathlib import Path

from collect.recorder import Recorder, MANIFEST_FILE


def test_recorder_creates_session_dir(tmp_path):
    rec = Recorder(label="wave", data_dir=str(tmp_path))
    sid = rec.start_session(participant="test_user")
    session_dir = tmp_path / sid
    assert session_dir.exists()

    meta_path = session_dir / "metadata.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["label"] == "wave"
    assert meta["participant"] == "test_user"
    rec.finish_session()


def test_recorder_saves_npy(tmp_path):
    rec = Recorder(label="swipe_left", data_dir=str(tmp_path))
    rec.start_session(participant="alice")

    window = np.ones(40, dtype=np.float32) * 42.0
    path = rec.record(window)

    assert path.exists()
    loaded = np.load(path)
    np.testing.assert_array_almost_equal(loaded, window)
    rec.finish_session()


def test_recorder_sequential_filenames(tmp_path):
    rec = Recorder(label="push", data_dir=str(tmp_path))
    rec.start_session()

    for _ in range(3):
        rec.record(np.zeros(40))

    sid = list(tmp_path.iterdir())[0].name
    files = sorted((tmp_path / sid).glob("push_*.npy"))
    assert len(files) == 3
    assert files[0].name == "push_0000.npy"
    assert files[2].name == "push_0002.npy"
    rec.finish_session()


def test_recorder_manifest_written(tmp_path, monkeypatch):
    manifest_path = tmp_path / "sessions.csv"
    monkeypatch.setattr("collect.recorder.MANIFEST_FILE", str(manifest_path))

    rec = Recorder(label="wave", data_dir=str(tmp_path / "raw"))
    rec.start_session(participant="bob")
    rec.record(np.zeros(40))
    rec.finish_session()

    assert manifest_path.exists()
    content = manifest_path.read_text()
    assert "wave" in content
    assert "bob" in content
