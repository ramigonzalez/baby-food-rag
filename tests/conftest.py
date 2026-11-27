import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources" / "manifest.yaml"
RAW = ROOT / "data" / "raw"


@pytest.fixture(autouse=True)
def _offline_and_keyless(monkeypatch):
    """Tests never read the owner's .env, never see a real key, and never call OpenRouter.

    Other URLs pass through: the download tests use a local HTTP server.
    """
    monkeypatch.setattr("baby_food_rag.cli.load_env", lambda *args, **kwargs: False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    real = urllib.request.urlopen

    def guarded(request, *args, **kwargs):
        url = getattr(request, "full_url", str(request))
        if "openrouter.ai" in url:
            raise AssertionError(f"a test tried to call {url}")
        return real(request, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)
