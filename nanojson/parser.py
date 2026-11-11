"""nanojson.parser -- a strict RFC 8259 JSON parser.

This module implements JSON deserialisation from scratch: a hand written
recursive-descent scanner over the input text.  The Python standard library
``json`` module is not used (nor imported) anywhere.

Public API
----------
``loads(text)``
    Deserialize a ``str`` containing a JSON document.
``load(fp)``
    Deserialize a file-like object whose ``read()`` returns ``str``.
``JSONParseError``
    ``ValueError`` subclass carrying ``lineno`` / ``colno`` / ``pos``.

Supported values: object, array, string (with all RFC 8259 escapes,
including ``\\uXXXX`` surrogate pairs), number (int and float, with
optional fraction and exponent), ``true``, ``false`` and ``null``.

The scanner is a recursive-descent parser, so each nesting level consumes
Python stack frames: documents nested deeper than roughly 300 levels raise
``RecursionError`` on CPython's default recursion limit of 1000 (the
pure-Python fallback scanner of the standard library behaves the same way;
the stdlib's C accelerator digs deeper).  This is a documented difference
from the stdlib, asserted nowhere on purpose -- the project requires and
tests 100 levels, which parse fine.
"""

__all__ = ["JSONParseError", "loads", "load"]

# RFC 8259 allows exactly these four characters as insignificant whitespace.
_WHITESPACE = " \t\n\r"

# Characters that must terminate a "bare" scan inside a string.
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_DIGITS = frozenset("0123456789")

# Two-character escape sequences (RFC 8259 section 7, "escape").
_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

_NUMBER_START = "-0123456789"


class JSONParseError(ValueError):
    """Raised when the input text is not a valid JSON document.

    Attributes
    ----------
    msg : str
        Human readable description of the problem.
    doc : str
        The text that was being parsed.
    pos : int
        Zero based character offset of the error in ``doc``.
    lineno : int
        One based line number of the error.
    colno : int
        One based column number of the error (counted from the line start).
    """

    def __init__(self, msg, doc="", pos=0, lineno=1, colno=1):
        super().__init__("%s: line %d column %d (char %d)" % (msg, lineno, colno, pos))
        self.msg = msg
        self.doc = doc
        self.pos = pos
        self.lineno = lineno
        self.colno = colno


def _position(doc, pos):
    """Return the (lineno, colno) of zero-based offset ``pos`` in ``doc``.

    Lines are counted from 1 and columns are counted from 1; a column points
    at the offending character itself (or one past the end of a line).
    """
    newline_index = doc.rfind("\n", 0, pos)
    if newline_index < 0:
        return 1, pos + 1
    lineno = doc.count("\n", 0, pos) + 1
    colno = pos - newline_index
    return lineno, colno


class _Parser:
    """Recursive-descent scanner implementing the RFC 8259 grammar."""

    def __init__(self, doc):
        self.doc = doc
        self.pos = 0
        self.end = len(doc)

    # ------------------------------------------------------------------ util

    def error(self, msg, pos=None):
        if pos is None:
            pos = self.pos
        lineno, colno = _position(self.doc, pos)
        raise JSONParseError(msg, self.doc, pos, lineno, colno)

    def skip_ws(self):
        doc = self.doc
        pos = self.pos
        end = self.end
        while pos < end and doc[pos] in _WHITESPACE:
            pos += 1
        self.pos = pos

    # ------------------------------------------------------------- top level

    def parse(self):
        self.skip_ws()
        if self.pos >= self.end:
            self.error("Expecting value")
        value = self.parse_value()
        self.skip_ws()
        if self.pos < self.end:
            self.error("Extra data")
        return value

    # ---------------------------------------------------------------- values

    def parse_value(self):
        doc = self.doc
        pos = self.pos
        if pos >= self.end:
            self.error("Expecting value")
        ch = doc[pos]
        if ch == "{":
            return self.parse_object()
        if ch == "[":
            return self.parse_array()
        if ch == '"':
            return self.parse_string()
        if ch in _NUMBER_START:
            return self.parse_number()
        if doc.startswith("true", pos):
            self.pos = pos + 4
            return True
        if doc.startswith("false", pos):
            self.pos = pos + 5
            return False
        if doc.startswith("null", pos):
            self.pos = pos + 4
            return None
        self.error("Expecting value")

    # --------------------------------------------------------------- objects

    def parse_object(self):
        self.pos += 1  # consume '{'
        self.skip_ws()
        if self.pos >= self.end:
            self.error("Expecting value")
        if self.doc[self.pos] == "}":
            self.pos += 1
            return {}
        obj = {}
        while True:
            self.skip_ws()
            if self.pos >= self.end or self.doc[self.pos] != '"':
                self.error("Expecting property name enclosed in double quotes")
            key = self.parse_string()
            self.skip_ws()
            if self.pos >= self.end or self.doc[self.pos] != ":":
                self.error("Expecting ':' delimiter")
            self.pos += 1
            self.skip_ws()
            value = self.parse_value()
            obj[key] = value
            self.skip_ws()
            if self.pos >= self.end:
                self.error("Expecting ',' delimiter")
            ch = self.doc[self.pos]
            if ch == "}":
                self.pos += 1
                return obj
            if ch != ",":
                self.error("Expecting ',' delimiter")
            self.pos += 1  # a trailing comma would fail on the next key check

    # ---------------------------------------------------------------- arrays

    def parse_array(self):
        self.pos += 1  # consume '['
        self.skip_ws()
        if self.pos >= self.end:
            self.error("Expecting value")
        if self.doc[self.pos] == "]":
            self.pos += 1
            return []
        array = []
        while True:
            self.skip_ws()
            array.append(self.parse_value())
            self.skip_ws()
            if self.pos >= self.end:
                self.error("Expecting ',' delimiter")
            ch = self.doc[self.pos]
            if ch == "]":
                self.pos += 1
                return array
            if ch != ",":
                self.error("Expecting ',' delimiter")
            self.pos += 1  # a trailing comma would fail on the next value check

    # --------------------------------------------------------------- strings

    def parse_string(self):
        doc = self.doc
        end = self.end
        start = self.pos  # index of the opening quote
        self.pos += 1
        begin = self.pos  # start of the current literal chunk
        chunks = []
        while True:
            if self.pos >= end:
                self.error("Unterminated string starting at", start)
            ch = doc[self.pos]
            if ch == '"':
                chunks.append(doc[begin:self.pos])
                self.pos += 1
                return "".join(chunks)
            if ch == "\\":
                chunks.append(doc[begin:self.pos])
                self.pos += 1
                if self.pos >= end:
                    self.error("Unterminated string starting at", start)
                esc = doc[self.pos]
                if esc in _ESCAPES:
                    chunks.append(_ESCAPES[esc])
                    self.pos += 1
                elif esc == "u":
                    self.pos += 1
                    code = self.parse_hex4()
                    if 0xD800 <= code <= 0xDBFF:
                        # High surrogate: pair it with a following low
                        # surrogate escape when one is present.
                        if (
                            self.pos + 1 < end
                            and doc[self.pos] == "\\"
                            and doc[self.pos + 1] == "u"
                        ):
                            saved = self.pos
                            self.pos += 2
                            low = self.parse_hex4()
                            if 0xDC00 <= low <= 0xDFFF:
                                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                            else:
                                # Not a low surrogate: keep the lone high one.
                                self.pos = saved
                    chunks.append(chr(code))
                else:
                    self.error("Invalid \\escape")
                begin = self.pos
                continue
            if ch < " ":
                self.error("Invalid control character")
            self.pos += 1

    def parse_hex4(self):
        doc = self.doc
        pos = self.pos
        chunk = doc[pos:pos + 4]
        if len(chunk) != 4:
            self.error("Invalid \\uXXXX escape")
        for ch in chunk:
            if ch not in _HEX_DIGITS:
                self.error("Invalid \\uXXXX escape")
        self.pos = pos + 4
        return int(chunk, 16)

    # --------------------------------------------------------------- numbers

    def parse_number(self):
        doc = self.doc
        end = self.end
        start = self.pos
        pos = self.pos

        if doc[pos] == "-":
            pos += 1
        if pos >= end or doc[pos] not in _DIGITS:
            self.error("Expecting value", start)

        # int part: '0' or [1-9][0-9]*
        if doc[pos] == "0":
            pos += 1
        else:
            while pos < end and doc[pos] in _DIGITS:
                pos += 1

        # frac part
        if pos < end and doc[pos] == ".":
            pos += 1
            if pos >= end or doc[pos] not in _DIGITS:
                self.error("Invalid number: expected digit after decimal point")
            while pos < end and doc[pos] in _DIGITS:
                pos += 1

        # exp part
        if pos < end and doc[pos] in "eE":
            pos += 1
            if pos < end and doc[pos] in "+-":
                pos += 1
            if pos >= end or doc[pos] not in _DIGITS:
                self.error("Invalid number: expected digit in exponent")
            while pos < end and doc[pos] in _DIGITS:
                pos += 1

        token = doc[start:pos]
        self.pos = pos
        if "." in token or "e" in token or "E" in token:
            return float(token)
        return int(token)


def loads(text):
    """Deserialize ``text`` (a ``str`` containing a JSON document).

    Raises :class:`JSONParseError` (a ``ValueError`` subclass) when the text
    is not valid RFC 8259 JSON, with ``lineno`` and ``colno`` attributes.
    """
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8")
    if isinstance(text, str):
        return _Parser(text).parse()
    raise TypeError(
        "the JSON object must be str or bytes, not %s" % type(text).__name__
    )


def load(fp):
    """Deserialize a JSON document read from the file-like object ``fp``.

    ``fp.read()`` must return a ``str`` (i.e. the file should be opened in
    text mode) or ``bytes``/``bytearray`` (decoded as UTF-8).
    """
    try:
        text = fp.read()
    except AttributeError:
        raise TypeError("load() expects a file-like object with read()") from None
    return loads(text)
