import gc

import pytest


@pytest.fixture(autouse=True)
def collect_renderers():
    """MuJoCo renderers own a GL context that must be freed on the thread that made it; collecting after every test
    keeps an unclosed one from being finalised later on a TestClient worker thread, which segfaults."""
    yield
    gc.collect()


@pytest.fixture(params=["menagerie", "procedural"])
def models(request, monkeypatch):
    """Both ways the factory is built: with the Menagerie models fetched, and on a fresh clone without them."""
    from airo import menagerie

    if request.param == "procedural":
        monkeypatch.setattr(menagerie, "ROOT", menagerie.ROOT / "missing")
    elif not menagerie.available(menagerie.UR5E, menagerie.GRIPPER, menagerie.G1):
        pytest.skip("Menagerie models not fetched: uv run scripts/fetch_models.py")
    return request.param
