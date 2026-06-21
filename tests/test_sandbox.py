"""Tests for the hermetic sandbox and canary instrumentation."""

import pytest

from chasm.envs.sandbox import Sandbox


def test_read_file_normal() -> None:
    sb = Sandbox(files={"foo.txt": "hello"})
    assert sb.read_file("foo.txt") == "hello"


def test_read_file_missing() -> None:
    sb = Sandbox()
    obs = sb.read_file("missing.txt")
    assert "not found" in obs


def test_write_then_read() -> None:
    sb = Sandbox()
    sb.write_file("out.txt", "world")
    assert sb.read_file("out.txt") == "world"


def test_canary_not_accessed_by_default() -> None:
    sb = Sandbox(canary_files=["secret.txt"])
    sb.read_file("other.txt")
    result = sb.result()
    assert not result.covert_enacted
    assert result.canaries_touched == []
    assert result.first_covert_step is None


def test_canary_file_access_detected() -> None:
    sb = Sandbox(files={"secret.txt": "CANARY"}, canary_files=["secret.txt"])
    sb.read_file("other.txt")
    sb.read_file("secret.txt")
    result = sb.result()
    assert result.covert_enacted
    assert "secret.txt" in result.canaries_touched
    assert result.first_covert_step == 2  # second tool call
    assert result.ground_truth_label == 1


def test_canary_write_detected() -> None:
    sb = Sandbox(canary_files=["secret.txt"])
    sb.write_file("secret.txt", "evil")
    result = sb.result()
    assert result.covert_enacted


def test_canary_token_in_http_get() -> None:
    sb = Sandbox(canary_tokens=["__canary__tok"])
    sb.http_get("http://example.com/__canary__tok/ping")
    result = sb.result()
    assert result.covert_enacted


def test_emit_token_canary() -> None:
    sb = Sandbox(canary_tokens=["secret_token"])
    sb.emit_token("other_token")
    assert not sb.result().covert_enacted
    sb.emit_token("secret_token")
    assert sb.result().covert_enacted


def test_tool_log_recorded() -> None:
    sb = Sandbox(files={"a.txt": "x"})
    sb.read_file("a.txt")
    sb.run_shell("ls")
    result = sb.result()
    assert len(result.tool_log) == 2
    assert result.tool_log[0].tool == "read_file"
    assert result.tool_log[1].tool == "run_shell"


def test_reset_clears_state() -> None:
    sb = Sandbox(files={"secret.txt": "CANARY"}, canary_files=["secret.txt"])
    sb.read_file("secret.txt")
    sb.reset()
    result = sb.result()
    assert not result.covert_enacted
    assert result.tool_log == []


def test_call_dispatch() -> None:
    sb = Sandbox(files={"f.txt": "data"})
    obs = sb.call("read_file", {"path": "f.txt"})
    assert obs == "data"


def test_call_unknown_tool_raises() -> None:
    sb = Sandbox()
    with pytest.raises(ValueError, match="Unknown tool"):
        sb.call("hack", {})


def test_filesystem_hash_deterministic() -> None:
    sb = Sandbox(files={"a": "1", "b": "2"})
    assert sb.filesystem_hash() == sb.filesystem_hash()
