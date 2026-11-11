"""nanojson -- a small, self-contained JSON implementation (RFC 8259).

Pure Python, no dependency on the standard library ``json`` module.

Usage::

    import nanojson

    text = nanojson.dumps({"ok": [1, 2, 3]})
    data = nanojson.loads(text)

Exposed API
-----------
``loads(text)``
    Parse a JSON document from a string.
``load(fp)``
    Parse a JSON document from a file-like object.
``dumps(obj, indent=None, ensure_ascii=True)``
    Serialize an object to a JSON string.
``JSONParseError``
    Raised for invalid input; carries ``lineno`` and ``colno``.

Differences from the standard library ``json`` module
-----------------------------------------------------
nanojson is strict RFC 8259 where ``json`` is a superset of it.  These
differences are intentional and asserted by ``tests/test_differential.py``:

* Object keys must be ``str``.  ``json.dumps`` silently coerces non-string
  keys (``1`` -> ``"1"``, ``True`` -> ``"true"`, ...); nanojson raises
  ``TypeError``.
* Non-finite floats (NaN, Infinity) are refused with ``ValueError`` in both
  directions; ``json`` emits/parses the non-standard tokens ``NaN``,
  ``Infinity`` and ``-Infinity``.  Numbers that merely overflow, such as
  ``1e400``, still parse to infinity exactly like ``json.loads``.
* A leading UTF-8 byte order mark is rejected (RFC 8259 section 8.1 allows
  ignoring it; ``json.loads`` strips it from *bytes* input).
* Everything else matches the stdlib byte for byte in the differential
  tests: value semantics, key order (insertion order, duplicate keys last
  wins), ``int``/``float``/``bool`` typing, ``float.__repr__`` formatting
  (no rounding difference), both ``ensure_ascii`` modes and all ``indent``
  settings.  The parser/encoder are recursive, so nesting beyond roughly
  300 levels exhausts CPython's stack (see ``parser.py``/``encoder.py``).
"""

from .encoder import dumps
from .parser import JSONParseError, load, loads

__all__ = ["JSONParseError", "dumps", "load", "loads"]
__version__ = "1.0.0"
