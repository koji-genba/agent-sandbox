#!/usr/bin/env python3
"""jsonx -- command line front end for the nanojson package.

nanojson is a self-contained, pure Python JSON implementation (RFC 8259)
that never imports the standard library ``json`` module; this CLI depends
only on ``argparse`` and ``sys``.

Subcommands
-----------
validate FILE
    Check that FILE contains a single valid JSON document.  Nothing is
    written to stdout on success.  On invalid input a diagnostic containing
    the line and column of the error is written to stderr.
    FILE may be ``-`` to read the document from stdin.

format FILE [--indent N] [--compact] [--no-ensure-ascii]
    Parse FILE and write it back pretty printed to stdout.  ``--indent N``
    sets the number of spaces per nesting level (default: 2), ``--compact``
    disables indentation entirely and ``--no-ensure-ascii`` emits non-ASCII
    characters literally instead of ``\\uXXXX`` escapes.

Exit codes
----------
0
    Success (document is valid / was formatted).
1
    The document is not valid JSON (message with line and column on
    stderr), or it cannot be serialized (e.g. it contains a number such as
    ``1e999999`` that parses to infinity, which JSON cannot represent).
2
    The file could not be read (missing, unreadable or not valid UTF-8), a
    bad option was given, or no subcommand was supplied.

Being strict by design, nanojson rejects documents the standard library
accepts: numbers like ``NaN``/``Infinity``, and a leading UTF-8 byte order
mark (see ``nanojson/__init__.py`` for the full list of differences).
"""

import argparse
import sys

import nanojson
from nanojson import JSONParseError

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_ERROR = 2

# Width of the source excerpt shown under a parse error message.
_SNIPPET_WIDTH = 96


def _read_document(path):
    """Return the text of ``path`` ("-" means stdin).  Raises OSError/ValueError."""
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()


def _stderr_safe(text):
    """Return ``text`` with characters that would break stderr replaced.

    Control characters are shown as ``?`` (a tab becomes a space so that the
    caret below stays roughly aligned) and characters outside the encoding
    stderr uses -- lone surrogates for instance -- are replaced too.
    """
    encoding = sys.stderr.encoding or "utf-8"
    out = []
    for ch in text:
        if ch == "\t":
            out.append(" ")
            continue
        try:
            ch.encode(encoding, errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            out.append("?")
        else:
            out.append(ch if ch.isprintable() else "?")
    return "".join(out)


def _snippet(doc, exc):
    """Return a two-line excerpt of ``doc`` around the error ``exc``.

    The second line carries a caret under the offending column.  Returns an
    empty string when the error position does not map onto a source line.
    """
    lines = doc.split("\n")
    if not 1 <= exc.lineno <= len(lines):
        return ""
    raw = lines[exc.lineno - 1]
    col = max(exc.colno - 1, 0)  # zero-based offset within the line
    if len(raw) <= _SNIPPET_WIDTH:
        window, offset = raw, 0
    else:
        offset = max(0, min(col - _SNIPPET_WIDTH // 4, len(raw) - _SNIPPET_WIDTH))
        window = raw[offset:offset + _SNIPPET_WIDTH]
    prefix = "    | "
    shown = _stderr_safe(window)
    caret = " " * max(col - offset, 0) + "^"
    return "%s%s\n%s%s" % (prefix, shown, prefix, caret)


def _report_parse_error(path, doc, exc):
    """Print a line/column diagnostic for ``exc`` to stderr."""
    sys.stderr.write(
        "jsonx: %s: line %d, column %d: %s\n"
        % (path, exc.lineno, exc.colno, exc.msg)
    )
    snippet = _snippet(doc, exc)
    if snippet:
        sys.stderr.write(snippet + "\n")


def cmd_validate(args):
    """Implement ``jsonx validate``: exit 0 for valid JSON, 1 otherwise."""
    try:
        doc = _read_document(args.file)
    except UnicodeDecodeError:
        sys.stderr.write(
            "jsonx: %s: file is not valid UTF-8, cannot check JSON\n" % (args.file,)
        )
        return EXIT_ERROR
    except OSError as exc:
        sys.stderr.write(
            "jsonx: %s: %s\n" % (args.file, exc.strerror or str(exc))
        )
        return EXIT_ERROR
    try:
        nanojson.loads(doc)
    except JSONParseError as exc:
        _report_parse_error(args.file, doc, exc)
        return EXIT_INVALID
    return EXIT_OK


def cmd_format(args):
    """Implement ``jsonx format``: pretty print the document to stdout."""
    indent = None if args.compact else args.indent
    if indent is not None and indent < 0:
        sys.stderr.write("jsonx: --indent must be >= 0, got %d\n" % (indent,))
        return EXIT_ERROR
    try:
        doc = _read_document(args.file)
    except UnicodeDecodeError:
        sys.stderr.write(
            "jsonx: %s: file is not valid UTF-8, cannot format\n" % (args.file,)
        )
        return EXIT_ERROR
    except OSError as exc:
        sys.stderr.write(
            "jsonx: %s: %s\n" % (args.file, exc.strerror or str(exc))
        )
        return EXIT_ERROR
    try:
        value = nanojson.loads(doc)
    except JSONParseError as exc:
        _report_parse_error(args.file, doc, exc)
        return EXIT_INVALID
    try:
        text = nanojson.dumps(
            value, indent=indent, ensure_ascii=not args.no_ensure_ascii
        )
    except ValueError as exc:
        # loads() can produce infinity (e.g. from "1e999999"), which has no
        # JSON representation; nanojson refuses to serialize it.
        sys.stderr.write("jsonx: %s: %s\n" % (args.file, exc))
        return EXIT_INVALID
    try:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stderr.write(
            "jsonx: %s: output cannot be encoded as %s; "
            "drop --no-ensure-ascii\n" % (args.file, sys.stdout.encoding or "ascii")
        )
        return EXIT_ERROR
    except OSError:
        # Unwritable destination (disk full, closed pipe, ...).
        return EXIT_ERROR
    return EXIT_OK


def build_parser():
    """Return the argument parser for the jsonx CLI."""
    parser = argparse.ArgumentParser(
        prog="jsonx",
        description="Validate and pretty-print JSON documents using nanojson.",
    )
    sub = parser.add_subparsers(
        dest="command", metavar="{validate,format}", required=True
    )

    p_validate = sub.add_parser(
        "validate", help="check that FILE is a valid JSON document"
    )
    p_validate.add_argument(
        "file",
        metavar="FILE",
        help="JSON document to check, or '-' for stdin",
    )

    p_format = sub.add_parser(
        "format", help="pretty-print FILE to stdout"
    )
    p_format.add_argument(
        "file",
        metavar="FILE",
        help="JSON document to format, or '-' for stdin",
    )
    p_format.add_argument(
        "--indent",
        type=int,
        default=2,
        metavar="N",
        help="spaces per nesting level (default: %(default)s)",
    )
    p_format.add_argument(
        "--compact",
        action="store_true",
        help="emit the whole document on one line",
    )
    p_format.add_argument(
        "--no-ensure-ascii",
        action="store_true",
        help="emit non-ASCII characters literally instead of \\uXXXX escapes",
    )
    return parser


def main(argv=None):
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        return cmd_validate(args)
    return cmd_format(args)


if __name__ == "__main__":
    sys.exit(main())
