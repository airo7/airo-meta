import importlib
from pathlib import Path

from airo import paths


def test_data_is_anchored_at_the_repo_and_overridable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIRO_DATA", raising=False)
    assert importlib.reload(paths).DATA == Path(__file__).resolve().parent.parent / "data"
    monkeypatch.setenv("AIRO_DATA", str(tmp_path))
    assert importlib.reload(paths).DATA == tmp_path
    monkeypatch.undo()
    importlib.reload(paths)
