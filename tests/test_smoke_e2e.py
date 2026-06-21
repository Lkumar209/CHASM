"""Smoke e2e test: verifies the scaffold runs without errors."""

import pytest


@pytest.mark.smoke
def test_scaffold_imports() -> None:
    """Ensure all top-level modules are importable."""
    import chasm
    import chasm.models
    import chasm.monitors
    import chasm.utils

    assert chasm.__version__ == "0.1.0"


@pytest.mark.smoke
def test_echo_backend_smoke() -> None:
    """Verify echo backend end-to-end with cache."""
    from chasm.models.base import BackendConfig
    from chasm.models.factory import build_backend

    cfg = BackendConfig(backend="echo")
    backend = build_backend(cfg)
    result = backend.generate("smoke test prompt", max_new_tokens=16)
    assert len(result.text) > 0
    acc = backend.accounting()
    assert acc["prompt_tokens"] >= 0


@pytest.mark.smoke
def test_seeding_smoke() -> None:
    from chasm.utils import seed_everything

    seed_everything(0)
    seed_everything(99)  # should not raise


@pytest.mark.smoke
def test_hashing_smoke() -> None:
    from chasm.utils import content_hash

    h = content_hash({"experiment": "smoke", "seed": 42})
    assert isinstance(h, str) and len(h) == 16
