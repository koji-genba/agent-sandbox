#!/usr/bin/env python3
"""Count word frequencies in a text file and print the top N words."""

import argparse
import re
import sys
from collections import Counter

WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def count_words(text):
    """Return a Counter of lowercase words found in text."""
    return Counter(m.group(0).lower() for m in WORD_RE.finditer(text))


def top_words(text, n):
    """Return the n most common (word, count) pairs, ties broken alphabetically."""
    counts = count_words(text)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def positive_int(value):
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}")
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1: {value}")
    return ivalue


def build_parser():
    parser = argparse.ArgumentParser(
        prog="wordfreq.py",
        description="Print the most frequent words in a text file "
        "(case-insensitive) as 'COUNT WORD' lines.",
    )
    parser.add_argument("file", help="path to the input text file")
    parser.add_argument(
        "--top",
        "-n",
        type=positive_int,
        default=10,
        metavar="N",
        help="number of words to show (default: 10)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"wordfreq.py: error: {e}", file=sys.stderr)
        return 1
    for word, count in top_words(text, args.top):
        print(f"{count} {word}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
