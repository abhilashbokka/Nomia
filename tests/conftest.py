import pytest


@pytest.fixture(autouse=True)
def _fresh_model_warmup_state():
    """The cold-start timeout logic keeps per-process state about which Ollama models have
    already answered once; tests must not leak that across each other."""
    from nomia import classify

    classify._WARMED_MODELS.clear()
    yield
    classify._WARMED_MODELS.clear()


@pytest.fixture(autouse=True)
def _no_real_siglip(monkeypatch):
    """Unit tests must never load the real SigLIP model: it's an optional dependency, takes
    seconds to load, and holds ~800MB. With it reported unavailable, classify_file falls
    through to the (mocked) VLM path, which is what almost every existing test exercises.
    Fast-path tests re-patch nomia.siglip.is_available/scores with controlled fakes."""
    monkeypatch.setattr("nomia.siglip.is_available", lambda: False)
