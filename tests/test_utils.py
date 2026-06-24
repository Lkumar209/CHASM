"""Unit tests for utils: seeding, hashing."""

from chasm.utils.hashing import content_hash
from chasm.utils.seeding import seed_everything


def test_seed_everything_deterministic() -> None:
    import random

    seed_everything(42)
    a = [random.random() for _ in range(5)]
    seed_everything(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_content_hash_stable() -> None:
    obj = {"a": 1, "b": [2, 3], "c": "hello"}
    assert content_hash(obj) == content_hash(obj)


def test_content_hash_order_invariant() -> None:
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert content_hash(a) == content_hash(b)


def test_content_hash_distinct() -> None:
    assert content_hash({"a": 1}) != content_hash({"a": 2})
