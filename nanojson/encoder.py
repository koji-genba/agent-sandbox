"""nanojson.encoder -- JSON serialisation.

Serialises Python objects to JSON text without using the standard library
``json`` module.

Supported types: ``int``, ``float``, ``str``, ``bool``, ``None``,
``list``, ``tuple`` and ``dict`` (keys must be ``str``).
Non-finite floats (NaN / Infinity) and circular references raise
``ValueError``; unsupported objects and non-string dict keys raise
``TypeError``.

The writer recurses, so structures nested deeper than roughly 300 levels
raise ``RecursionError`` on CPython's default recursion limit (documented
difference from the stdlib, whose C encoder digs deeper; the project
requires and tests 100 levels).

Unlike ``json.dumps``, non-string dict keys are *not* coerced to strings
and floats that are not finite are *not* emitted as ``NaN``/``Infinity``
-- both are refused, because RFC 8259 has no representation for them.
Float formatting itself is identical to the stdlib: ``float.__repr__``,
the shortest decimal string that round-trips, so there is no rounding
difference.
"""

__all__ = ["dumps"]

_INF = float("inf")

# Escape table: the mandatory escapes plus every C0 control character.
_ESCAPE_DCT = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}
for _i in range(0x20):
    _ESCAPE_DCT.setdefault(chr(_i), "\\u%04x" % _i)
del _i

# Unescaped ASCII characters run from 0x20 (space) up to 0x7e (tilde).
_ASCII_MAX = 0x7E


def _encode_basestring(s, ensure_ascii):
    """Return ``s`` as a JSON string literal, quotes included."""
    if ensure_ascii:
        if s.isascii() and s.isprintable() and '"' not in s and "\\" not in s:
            return '"' + s + '"'
        chunks = ['"']
        for ch in s:
            code = ord(ch)
            if 0x20 <= code <= _ASCII_MAX:
                if code == 0x22 or code == 0x5C:  # '"' or '\'
                    chunks.append(_ESCAPE_DCT[ch])
                else:
                    chunks.append(ch)
            else:
                esc = _ESCAPE_DCT.get(ch)
                if esc is not None:
                    chunks.append(esc)
                elif code > 0xFFFF:
                    # Non-BMP characters are emitted as a UTF-16 surrogate pair.
                    code -= 0x10000
                    high = 0xD800 + (code >> 10)
                    low = 0xDC00 + (code & 0x3FF)
                    chunks.append("\\u%04x\\u%04x" % (high, low))
                else:
                    chunks.append("\\u%04x" % code)
        chunks.append('"')
        return "".join(chunks)
    # ensure_ascii == False: emit non-ASCII characters literally.
    chunks = ['"']
    for ch in s:
        esc = _ESCAPE_DCT.get(ch)
        chunks.append(esc if esc is not None else ch)
    chunks.append('"')
    return "".join(chunks)


def _floatstr(value):
    if value != value:  # NaN
        raise ValueError("Out of range float values are not JSON compliant: nan")
    if value == _INF or value == -_INF:
        raise ValueError(
            "Out of range float values are not JSON compliant: %r" % (value,)
        )
    return float.__repr__(value)


class _Writer:
    """Builds the JSON text, guarding against circular references."""

    def __init__(self, indent, ensure_ascii):
        self.indent = indent
        self.ensure_ascii = ensure_ascii
        self.markers = set()

    def write(self, obj, level):
        """Return the JSON text for ``obj``; ``level`` is its indent prefix."""
        if obj is None:
            return "null"
        if obj is True:
            return "true"
        if obj is False:
            return "false"
        if isinstance(obj, str):
            return _encode_basestring(obj, self.ensure_ascii)
        if isinstance(obj, bool):  # bool is an int subclass: test it first
            return "true" if obj else "false"
        if isinstance(obj, int):
            return int.__repr__(obj)
        if isinstance(obj, float):
            return _floatstr(obj)
        if isinstance(obj, (list, tuple)):
            return self.write_array(obj, level)
        if isinstance(obj, dict):
            return self.write_dict(obj, level)
        raise TypeError(
            "Object of type %s is not JSON serializable" % type(obj).__name__
        )

    def write_array(self, obj, level):
        if id(obj) in self.markers:
            raise ValueError("Circular reference detected")
        self.markers.add(id(obj))
        try:
            if self.indent is None:
                return "[" + ", ".join(
                    self.write(item, level) for item in obj
                ) + "]"
            if len(obj) == 0:
                return "[]"
            inner = level + self.indent
            return "[\n" + ",\n".join(
                inner + self.write(item, inner) for item in obj
            ) + "\n" + level + "]"
        finally:
            self.markers.discard(id(obj))

    def write_dict(self, obj, level):
        if id(obj) in self.markers:
            raise ValueError("Circular reference detected")
        self.markers.add(id(obj))
        try:
            if len(obj) == 0:
                return "{}"
            if self.indent is None:
                pairs = []
                for key, value in obj.items():
                    if not isinstance(key, str):
                        raise TypeError(
                            "keys must be str, not %s" % type(key).__name__
                        )
                    pairs.append(
                        _encode_basestring(key, self.ensure_ascii)
                        + ": "
                        + self.write(value, level)
                    )
                return "{" + ", ".join(pairs) + "}"
            inner = level + self.indent
            lines = []
            for key, value in obj.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "keys must be str, not %s" % type(key).__name__
                    )
                lines.append(
                    inner
                    + _encode_basestring(key, self.ensure_ascii)
                    + ": "
                    + self.write(value, inner)
                )
            return "{\n" + ",\n".join(lines) + "\n" + level + "}"
        finally:
            self.markers.discard(id(obj))


def _normalize_indent(indent):
    if indent is None:
        return None
    if isinstance(indent, bool):
        indent = int(indent)
    if isinstance(indent, int):
        if indent < 0:
            raise TypeError(
                "'indent' must be None, a non-negative int or str, not int"
            )
        return " " * indent
    if isinstance(indent, str):
        return indent
    raise TypeError(
        "'indent' must be None, a non-negative int or str, not %s"
        % type(indent).__name__
    )


def dumps(obj, indent=None, ensure_ascii=True):
    """Serialize ``obj`` to a JSON formatted ``str``.

    Parameters
    ----------
    obj :
        ``None``, ``bool``, ``int``, ``float``, ``str``, ``list``, ``tuple``
        or ``dict``; nested containers are supported, ``dict`` keys must be
        ``str``.
    indent : int or str or None
        When not ``None`` the output is pretty printed with newlines and this
        indentation per nesting level (an ``int`` expands to that many spaces,
        a ``str`` is used verbatim).  Empty containers stay on a single line.
    ensure_ascii : bool
        ``True`` (the default) escapes every non-ASCII character as ``\\uXXXX``
        (non-BMP characters as surrogate pairs); ``False`` emits them
        literally.
    """
    indent = _normalize_indent(indent)
    return _Writer(indent, ensure_ascii).write(obj, "")
