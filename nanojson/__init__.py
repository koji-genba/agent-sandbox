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
"""

from .encoder import dumps
from .parser import JSONParseError, load, loads

__all__ = ["JSONParseError", "dumps", "load", "loads"]
__version__ = "1.0.0"
