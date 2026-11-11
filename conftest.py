"""Pytest configuration: make the repository root importable.

Keeps ``import nanojson`` working no matter how pytest is invoked (the
``pytest`` console script does not put the current directory on sys.path,
while ``python -m pytest`` does).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
