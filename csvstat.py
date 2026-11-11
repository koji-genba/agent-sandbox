#!/usr/bin/env python3
"""csvstat.py - CSV column statistics summary (Python 3 standard library only).

Reads a CSV file and prints one summary line per column:

    <column name>: non-empty=<n>, missing=<m>, unique=<u>

where

    non-empty  number of cells that contain a non-blank value
    missing    number of blank cells (empty string or whitespace-only,
               including cells absent because the row was shorter
               than the header)
    unique     number of distinct non-empty values in the column

Usage:
    python3 csvstat.py FILE

Exit codes:
    0  success (an empty CSV file prints nothing and is not an error)
    1  the file could not be read or parsed
    2  command-line usage error (argparse)
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass

PROG = "csvstat"


@dataclass
class ColumnStats:
    """Statistics collected for a single CSV column."""

    name: str
    non_empty: int = 0
    missing: int = 0
    unique: int = 0


def summarize(header: list[str], rows: list[list[str]]) -> list[ColumnStats]:
    """Compute per-column statistics from a header row and data rows.

    Columns are defined by ``header``.  Cells beyond the header length are
    ignored; rows shorter than the header contribute missing cells for the
    columns they do not have.  A cell is *missing* when it is empty or
    whitespace-only.  ``unique`` counts distinct non-empty values.
    """
    stats = [ColumnStats(name=name) for name in header]
    seen: list[set[str]] = [set() for _ in header]

    for row in rows:
        for i, stat in enumerate(stats):
            cell = row[i] if i < len(row) else ""
            if cell.strip() == "":
                stat.missing += 1
            else:
                stat.non_empty += 1
                seen[i].add(cell)

    for stat, values in zip(stats, seen):
        stat.unique = len(values)
    return stats


def read_csv(path: str) -> tuple[list[str], list[list[str]]]:
    """Read a CSV file and return (header, data rows).

    The file is read with the ``csv`` module using UTF-8 (a leading BOM is
    stripped).  Raises ``OSError`` / ``csv.Error`` on failure.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        rows = [row for row in reader]
    return header, rows


def format_stats(stats: list[ColumnStats]) -> str:
    """Render one summary line per column."""
    return "\n".join(
        f"{stat.name}: non-empty={stat.non_empty}, "
        f"missing={stat.missing}, unique={stat.unique}"
        for stat in stats
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Print per-column statistics (non-empty / missing / "
                    "unique counts) for a CSV file.",
    )
    parser.add_argument("file", help="path to the CSV file to summarize")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        header, rows = read_csv(args.file)
    except OSError as exc:
        print(f"{PROG}: error: cannot read {args.file!r}: {exc}", file=sys.stderr)
        return 1
    except csv.Error as exc:
        print(f"{PROG}: error: cannot parse {args.file!r}: {exc}", file=sys.stderr)
        return 1

    output = format_stats(summarize(header, rows))
    if output:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
