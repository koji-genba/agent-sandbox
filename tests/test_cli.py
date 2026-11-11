"""End-to-end tests for the ``jsonx.py`` command line interface.

Every test runs ``jsonx.py`` in a subprocess so that exit codes, stdout and
stderr are exercised exactly as a user sees them.  The standard library
``json`` module is *not* imported here: using it in tests is reserved for
``tests/test_differential.py`` (project rule).

Expected exit codes (documented in ``jsonx.py``):
    0 = success, 1 = invalid JSON document, 2 = usage/IO/encoding error.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONX = os.path.join(ROOT, "jsonx.py")


def run_jsonx(*args, stdin_text=None):
    """Run the CLI in a subprocess; returns the CompletedProcess."""
    env = dict(os.environ)
    # Deterministic byte-level assertions regardless of the ambient locale.
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, JSONX] + list(args),
        input=stdin_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=ROOT,
        timeout=60,
    )


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------


VALID_DOCUMENTS = [
    "{}",
    "[]",
    "null",
    "true",
    "42",
    "-0.5e3",
    '"text with \\"escapes\\" and \\\\ backslash"',
    '{"a": [1, 2, {"b": null}], "c": []}',
    '[\n  1,\n\t2\r\n]',
    '{"deep": ' + "[" * 100 + "1" + "]" * 100 + "}",
    '{"nul": "\\u0000", "pair": "\\ud83d\\ude00", "big": 123456789012345678901234567890}',
]


@pytest.mark.parametrize("document", VALID_DOCUMENTS)
def test_validate_accepts_valid_document(document, tmp_path):
    path = tmp_path / "valid.json"
    path.write_text(document, encoding="utf-8")
    proc = run_jsonx("validate", str(path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert proc.stderr == ""


def test_validate_accepts_unicode_document(tmp_path):
    path = tmp_path / "unicode.json"
    path.write_text('{"note": "café", "emoji": "\U0001f600"}', encoding="utf-8")
    proc = run_jsonx("validate", str(path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


@pytest.mark.parametrize(
    "document,line,column",
    [
        ('{"a": tru}', "line 1", "column 7"),
        ("{\n  \"a\": 1,\n  \"b\": oops\n}", "line 3", "column 8"),
        ("[1, 2", "line 1", "column 6"),
        ("", "line 1", "column 1"),
        ("   \n  ", "line 2", "column 3"),
        ("{\"a\": 1,}", "line 1", "column 9"),
        ('"unterminated', "line 1", "column 1"),  # points at the opening quote
    ],
)
def test_validate_rejects_invalid_document_with_position(
    document, line, column, tmp_path
):
    path = tmp_path / "invalid.json"
    path.write_text(document, encoding="utf-8")
    proc = run_jsonx("validate", str(path))
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert line in proc.stderr and column in proc.stderr
    assert "invalid.json" in proc.stderr  # the offending file is named


def test_validate_error_message_shows_the_offending_line(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"a": 1,\n  "b": tru\n}', encoding="utf-8")
    proc = run_jsonx("validate", str(path))
    assert proc.returncode == 1
    stderr_lines = proc.stderr.strip().splitlines()
    assert stderr_lines[0].startswith("jsonx:")
    assert "line 2, column 8" in stderr_lines[0]
    # The second and third lines quote the source with a caret pointer.
    assert '"b": tru' in stderr_lines[1]
    assert stderr_lines[2].strip().endswith("^")


def test_validate_reads_stdin_when_file_is_dash():
    assert run_jsonx("validate", "-", stdin_text="[1, 2, 3]").returncode == 0
    proc = run_jsonx("validate", "-", stdin_text="{")
    assert proc.returncode == 1
    assert "line 1" in proc.stderr and "column 2" in proc.stderr


def test_validate_missing_file_exits_2():
    proc = run_jsonx("validate", str(tmp_path := "/nonexistent/dir/x.json"))
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "x.json" in proc.stderr


def test_validate_non_utf8_file_exits_2(tmp_path):
    path = tmp_path / "binary.json"
    path.write_bytes(b'{"a": "\xff\xfe"}')
    proc = run_jsonx("validate", str(path))
    assert proc.returncode == 2
    assert "UTF-8" in proc.stderr


def test_validate_directory_instead_of_file_exits_2(tmp_path):
    proc = run_jsonx("validate", str(tmp_path))
    assert proc.returncode == 2


# --------------------------------------------------------------------------
# format
# --------------------------------------------------------------------------


def test_format_pretty_prints_with_default_indent_two():
    proc = run_jsonx("format", "-", stdin_text='{"a":[1,2],"b":{"c":null}}')
    assert proc.returncode == 0
    assert proc.stdout == (
        '{\n  "a": [\n    1,\n    2\n  ],\n  "b": {\n    "c": null\n  }\n}\n'
    )


def test_format_indent_zero_keeps_newlines_but_no_spaces():
    proc = run_jsonx("format", "-", "--indent", "0", stdin_text='[1,{"a":[]}]')
    assert proc.returncode == 0
    assert proc.stdout == '[\n1,\n{\n"a": []\n}\n]\n'


def test_format_indent_four():
    proc = run_jsonx("format", "-", "--indent", "4", stdin_text='[1,[2,3]]')
    assert proc.returncode == 0
    assert proc.stdout == '[\n    1,\n    [\n        2,\n        3\n    ]\n]\n'


def test_format_compact_flag_disables_indentation():
    proc = run_jsonx(
        "format", "-", "--compact", stdin_text='{ "a" : [ 1 , 2 ] }'
    )
    assert proc.returncode == 0
    assert proc.stdout == '{"a": [1, 2]}\n'


def test_format_preserves_key_order_and_duplicates_are_last_wins():
    proc = run_jsonx("format", "-", "--compact", stdin_text='{"z":1,"a":2,"a":3}')
    assert proc.returncode == 0
    assert proc.stdout == '{"z": 1, "a": 3}\n'


def test_format_escapes_non_ascii_by_default(tmp_path):
    path = tmp_path / "unicode.json"
    path.write_text('{"note": "café \U0001f600"}', encoding="utf-8")
    proc = run_jsonx("format", str(path))
    assert proc.returncode == 0
    assert proc.stdout == (
        '{\n  "note": "caf\\u00e9 \\ud83d\\ude00"\n}\n'
    )
    compact = run_jsonx("format", str(path), "--compact")
    assert compact.returncode == 0
    assert compact.stdout == '{"note": "caf\\u00e9 \\ud83d\\ude00"}\n'


def test_format_no_ensure_ascii_emits_unicode_literally(tmp_path):
    path = tmp_path / "unicode.json"
    path.write_text('{"note": "café"}', encoding="utf-8")
    proc = run_jsonx("format", str(path), "--no-ensure-ascii", "--compact")
    assert proc.returncode == 0
    assert proc.stdout == '{"note": "café"}\n'


def test_format_roundtrips_its_own_output(tmp_path):
    source = tmp_path / "in.json"
    target = tmp_path / "out.json"
    source.write_text('{"a":[1,2,{"b":null}],"c":{}}', encoding="utf-8")
    first = run_jsonx("format", str(source))
    assert first.returncode == 0
    target.write_text(first.stdout, encoding="utf-8")
    second = run_jsonx("format", str(target), "--compact")
    assert second.returncode == 0
    assert second.stdout == '{"a": [1, 2, {"b": null}], "c": {}}\n'


def test_format_rejects_invalid_document_with_position(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"a": 1,\n  "b": }', encoding="utf-8")
    proc = run_jsonx("format", str(path))
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "line 2" in proc.stderr and "column 8" in proc.stderr


def test_format_document_with_infinity_exits_1():
    # 1e999999 parses to infinity, which JSON cannot serialise.
    proc = run_jsonx("format", "-", stdin_text='{"x": 1e999999}')
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "float" in proc.stderr or "range" in proc.stderr


def test_format_missing_file_exits_2():
    proc = run_jsonx("format", "/nonexistent/dir/x.json")
    assert proc.returncode == 2
    assert proc.stdout == ""


def test_format_rejects_negative_indent():
    proc = run_jsonx("format", "-", "--indent", "-2", stdin_text="{}")
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "--indent" in proc.stderr


def test_format_non_utf8_file_exits_2(tmp_path):
    path = tmp_path / "binary.json"
    path.write_bytes(b'{"a": "\xff"}')
    proc = run_jsonx("format", str(path))
    assert proc.returncode == 2
    assert proc.stdout == ""


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------


def test_jsonx_is_executable_and_has_a_shebang():
    assert os.access(JSONX, os.X_OK), "jsonx.py should be executable"
    with open(JSONX, "r", encoding="utf-8") as fp:
        assert fp.readline().strip() == "#!/usr/bin/env python3"
    proc = subprocess.run(
        [JSONX, "validate", "-"],
        input="{}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
        timeout=60,
    )
    assert proc.returncode == 0  # direct ./jsonx.py execution works


def test_help_lists_both_subcommands():
    proc = run_jsonx("--help")
    assert proc.returncode == 0
    assert "validate" in proc.stdout and "format" in proc.stdout


def test_usage_errors_exit_2():
    assert run_jsonx().returncode == 2  # no subcommand
    assert run_jsonx("frobnicate").returncode == 2  # unknown subcommand
    assert run_jsonx("validate").returncode == 2  # missing FILE
    assert run_jsonx("validate", "--nope", "-").returncode == 2


def test_jsonx_never_imports_the_stdlib_json_module():
    """Guards the project rule: only tests/test_differential.py may use json."""
    probe = (
        "import sys; sys.path.insert(0, %r); import jsonx; "
        "print('json' in sys.modules, 'nanojson' in sys.modules)" % (ROOT,)
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False True"
