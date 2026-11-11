import subprocess
import sys
from pathlib import Path

import pytest

import wordfreq

SCRIPT = Path(__file__).with_name("wordfreq.py")


def write(tmp_path, content, name="input.txt"):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_normal_counts_and_lowercases(tmp_path, capsys):
    path = write(tmp_path, "The cat and the hat.\nTHE CAT sat!\n")
    assert wordfreq.main([path]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["3 the", "2 cat", "1 and", "1 hat", "1 sat"]


def test_empty_file_prints_nothing(tmp_path, capsys):
    path = write(tmp_path, "")
    assert wordfreq.main([path]) == 0
    assert capsys.readouterr().out == ""


def test_top_option_limits_output(tmp_path, capsys):
    path = write(tmp_path, "a a a b b c d e")
    assert wordfreq.main([path, "--top", "2"]) == 0
    assert capsys.readouterr().out.splitlines() == ["3 a", "2 b"]


def test_default_top_is_10(tmp_path, capsys):
    words = " ".join(f"w{i:02d}" for i in range(15))
    path = write(tmp_path, words)
    wordfreq.main([path])
    assert len(capsys.readouterr().out.splitlines()) == 10


@pytest.mark.parametrize("bad", ["0", "-1", "abc"])
def test_invalid_top_rejected(tmp_path, bad):
    path = write(tmp_path, "x")
    with pytest.raises(SystemExit) as exc:
        wordfreq.main([path, "--top", bad])
    assert exc.value.code == 2


def test_missing_file_returns_error(tmp_path, capsys):
    assert wordfreq.main([str(tmp_path / "nope.txt")]) == 1
    assert "error" in capsys.readouterr().err


def test_help_via_cli():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--top" in result.stdout
