import gc

import pytest


@pytest.fixture(autouse=True)
def collect_renderers():
    """MuJoCo renderers own a GL context that must be freed on the thread that made it; collecting after every test
    keeps an unclosed one from being finalised later on a TestClient worker thread, which segfaults."""
    yield
    gc.collect()
