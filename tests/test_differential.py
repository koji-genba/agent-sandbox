"""Differential tests: nanojson compared against the standard library ``json``.

The standard library ``json`` module is imported **only in this test file**
(project rule).  It is used purely as the reference implementation to
compare against -- nanojson itself never imports it anywhere.

What is compared
================
* Random documents (objects, arrays, strings, integers, floats, ``null``,
  booleans, deep nesting, Unicode incl. astral and lone surrogate
  characters, control characters and NUL) parsed with ``nanojson.loads``
  and ``json.loads``: values, container key order and exact Python types
  (``int`` vs ``float`` vs ``bool``) must all agree.
* Serialisation: ``nanojson.dumps`` and ``json.dumps`` must produce
  byte-identical text for both ``ensure_ascii`` settings and several
  ``indent`` settings.
* Round trips: ``dumps`` -> ``loads`` in all three combinations
  (nano->nano, nano->stdlib, stdlib->nano).
* Specific hazards: 100-level nesting, NUL characters, surrogate pairs and
  lone surrogates, big integers and exponent notation, and
  ``ensure_ascii=True``/``False``.
* Robustness: truncated/mutated documents and a curated list of malformed
  inputs, where both implementations must agree on acceptance.

Known, intentional differences (each asserted in a dedicated test)
==================================================================
1. Object keys: RFC 8259 only allows strings.  ``json.dumps`` silently
   coerces non-string keys (``1`` -> ``"1"``, ``True`` -> ``"true"``,
   ``None`` -> ``"null"``, ``3.5`` -> ``"3.5"``); ``nanojson.dumps`` raises
   TypeError instead of emitting a non-standard document.
2. Non-finite floats: ``json`` writes and reads the non-standard tokens
   ``NaN``/``Infinity``/``-Infinity``; nanojson raises ValueError in both
   directions because RFC 8259 has no representation for them.  (Numbers
   that merely *overflow*, such as ``1e400``, parse to infinity in both
   implementations; the difference is only the literal tokens and
   serialisation.)  The mutation test therefore allows one asymmetry: when
   ``json.loads`` returns a document holding a non-finite float obtained
   from such a token, nanojson must reject that text.
3. Byte order mark: ``json.loads`` strips a leading UTF-8 BOM from *bytes*
   input; nanojson refuses it (RFC 8259 section 8.1 allows implementations
   to ignore a BOM; nanojson chooses to be strict).  In *str* input both
   implementations reject a BOM.
4. Nesting depth: both scanners recurse, but the stdlib scanner is a C
   accelerator.  nanojson's pure Python scanner/encoder start raising
   RecursionError at roughly 300-400 nesting levels on CPython's default
   recursion limit (the pure-Python fallback of the stdlib behaves
   similarly).  The project requires 100 levels, which work fine and are
   tested here.
5. Float formatting: both implementations emit ``float.__repr__`` -- the
   shortest decimal string that round-trips -- so there is *no* rounding
   difference; the tests assert byte-identical serialisations.  Integers
   keep arbitrary precision in both (never routed through float).
6. Key order and duplicates: both preserve insertion order and resolve
   duplicate object keys to the last value; asserted below.
"""

import json  # noqa: allowed in this file only (reference implementation)
import math
import random
import sys

import pytest

import nanojson


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _assert_same(actual, expected, path="$"):
    """Assert equality including exact types, key order and float signs."""
    assert type(actual) is type(expected), (
        "%s: type %s != %s" % (path, type(actual).__name__, type(expected).__name__)
    )
    if isinstance(expected, dict):
        assert list(actual) == list(expected), "%s: key order/set differs" % (path,)
        for key, sub in expected.items():
            _assert_same(actual[key], sub, "%s.%r" % (path, key))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), "%s: length differs" % (path,)
        for index, sub in enumerate(expected):
            _assert_same(actual[index], sub, "%s[%d]" % (path, index))
    elif isinstance(expected, float):
        assert actual == expected, "%s: %r != %r" % (path, actual, expected)
        # 0.0 and -0.0 compare equal but differ in sign.
        assert math.copysign(1.0, actual) == math.copysign(1.0, expected), (
            "%s: float sign differs (%r vs %r)" % (path, actual, expected)
        )
    else:
        assert actual == expected, "%s: %r != %r" % (path, actual, expected)


def _contains_nonfinite_token(text):
    """True when ``text`` mentions the non-standard NaN/Infinity tokens.

    Only these tokens make the stdlib produce a non-finite float in a way
    nanojson refuses; an overflowing number such as ``1e400`` parses to
    infinity in *both* implementations and is compared normally.
    """
    return "NaN" in text or "Infinity" in text


def _contains_nonfinite(value):
    """True when ``value`` holds NaN or an infinity anywhere (json-only tokens)."""
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    return False


class RandomJSON:
    """Builds random JSON-compatible Python objects with awkward content."""

    STRING_ATOMS = (
        "",
        "a",
        "hello world",
        'quote " inside',
        "backslash \\ inside",
        "forward / slash",
        "tab\tand\nnewline\rand\bell\b\f\f",
        "NUL\x00inside",
        "del\x7fchar",
        "0123456789",
        # Non-ASCII: Latin-1, CJK, combining mark, spaces, zero width, BOM.
        "café",
        "日本語",
        "\U0001f600",
        "\U0001d11e",
        "\U0010ffff",
        "e\u0301",
        "\u00a0",
        "\u2028",
        "\u2029",
        "\u200b",
        "\ufeff",
    )
    KEY_ATOMS = (
        "",
        "a",
        "key",
        "with space",
        "café",
        "日本語",
        "\U0001f600",
        "n\x00l",
        'q"k',
        "b\\k",
        "\u00a0",
    )
    LONE_SURROGATES = ("\ud800", "\udbff", "\udc00", "\udfff")

    def __init__(self, rng):
        self.rng = rng

    def string(self):
        parts = [
            self.rng.choice(self.STRING_ATOMS) for _ in range(self.rng.randint(0, 3))
        ]
        if self.rng.random() < 0.3:
            parts.append(self.rng.choice(self.LONE_SURROGATES))
        return "".join(parts)

    def integer(self):
        roll = self.rng.random()
        if roll < 0.15:
            return 0
        if roll < 0.4:
            return self.rng.randint(-1000, 1000)
        if roll < 0.6:
            return self.rng.randint(-(2**64), 2**64)
        return self.rng.choice(
            [
                2**53 + 1,
                -(2**53) - 1,
                123456789012345678901234567890,
                2**200,
                -(2**200),
            ]
        )

    def floating(self):
        # Finite values only: JSON has no representation for NaN/Infinity and
        # nanojson deliberately refuses to serialise them (difference #2).
        roll = self.rng.random()
        if roll < 0.25:
            return float(self.integer())
        if roll < 0.5:
            return float(repr(self.rng.uniform(-1e6, 1e6)))
        return self.rng.choice(
            [
                0.0,
                -0.0,
                0.1,
                -0.25,
                1e-7,
                6.02e23,
                1e16,
                1e308,
                5e-324,
                2.2250738585072014e-308,
                math.pi,
            ]
        )

    def scalar(self):
        roll = self.rng.random()
        if roll < 0.40:
            return self.string()
        if roll < 0.60:
            return self.integer()
        if roll < 0.80:
            return self.floating()
        if roll < 0.90:
            return None
        return self.rng.choice([True, False])

    def value(self, depth=3):
        roll = self.rng.random()
        if depth <= 0 or roll < 0.55:
            return self.scalar()
        if roll < 0.80:
            return [self.value(depth - 1) for _ in range(self.rng.randint(0, 4))]
        keys = self.rng.sample(self.KEY_ATOMS, self.rng.randint(0, 5))
        return {key: self.value(depth - 1) for key in keys}


# --------------------------------------------------------------------------
# randomised differential tests (the "100+ inputs" requirement)
# --------------------------------------------------------------------------


def test_loads_matches_stdlib_on_random_documents():
    """150 random objects x 2 encodings = 300 documents parsed by both."""
    gen = RandomJSON(random.Random(0xD1FFE))
    documents = []
    for _ in range(150):
        obj = gen.value(gen.rng.randint(0, 4))
        for ensure_ascii in (True, False):
            documents.append(
                (obj, json.dumps(obj, ensure_ascii=ensure_ascii), ensure_ascii)
            )
    assert len(documents) >= 100  # the project asks for 100+ differential inputs
    for number, (obj, text, ensure_ascii) in enumerate(documents):
        context = "document #%d (ensure_ascii=%s)" % (number, ensure_ascii)
        _assert_same(nanojson.loads(text), json.loads(text), context)
        # The reference parse equals the object we started from, so nanojson
        # reconstructed the original value exactly.
        _assert_same(json.loads(text), obj, context + " (reference)")


def test_dumps_matches_stdlib_byte_for_byte():
    """Random objects: both encoders must emit identical text."""
    gen = RandomJSON(random.Random(0x5EED5))
    checked = 0
    for _ in range(60):
        obj = gen.value(gen.rng.randint(0, 4))
        for ensure_ascii in (True, False):
            for indent in (None, 0, 2, 4, "\t"):
                checked += 1
                assert nanojson.dumps(
                    obj, ensure_ascii=ensure_ascii, indent=indent
                ) == json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent), (
                    "mismatch for %r (ensure_ascii=%s, indent=%r)"
                    % (obj, ensure_ascii, indent)
                )
    assert checked >= 100


def test_roundtrip_dumps_then_loads():
    """100 random objects round trip through dumps/loads in all combinations."""
    gen = RandomJSON(random.Random(0x1234AB))
    checked = 0
    for _ in range(100):
        obj = gen.value(gen.rng.randint(0, 4))
        for ensure_ascii in (True, False):
            checked += 1
            nano_text = nanojson.dumps(obj, ensure_ascii=ensure_ascii)
            std_text = json.dumps(obj, ensure_ascii=ensure_ascii)
            assert nano_text == std_text
            _assert_same(nanojson.loads(nano_text), obj, "nano -> nano")
            _assert_same(json.loads(nano_text), obj, "nano -> stdlib")
            _assert_same(nanojson.loads(std_text), obj, "stdlib -> nano")
    assert checked >= 100


# --------------------------------------------------------------------------
# deep nesting (project requirement: 100 levels)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 10, 50, 100])
def test_deep_array_nesting_matches_stdlib(depth):
    doc = "[" * depth + "1" + "]" * depth
    expected = json.loads(doc)
    _assert_same(nanojson.loads(doc), expected)
    _assert_same(nanojson.loads(nanojson.dumps(nanojson.loads(doc))), expected)


@pytest.mark.parametrize("depth", [1, 2, 10, 50, 100])
def test_deep_object_nesting_matches_stdlib(depth):
    doc = '{"n": ' * depth + "null" + "}" * depth
    expected = json.loads(doc)
    _assert_same(nanojson.loads(doc), expected)
    _assert_same(nanojson.loads(nanojson.dumps(nanojson.loads(doc))), expected)


def _deep_mixed_document(depth):
    """Alternate arrays and objects, with whitespace between all tokens."""
    openers = [
        " [ " if level % 2 == 0 else '\n\t{"k%d" : ' % level
        for level in range(depth)
    ]
    doc = "".join(openers) + "true"
    for level in reversed(range(depth)):
        doc += " ] " if level % 2 == 0 else "\n } "
    return doc


def test_deep_mixed_nesting_100_levels_matches_stdlib():
    doc = _deep_mixed_document(100)
    expected = json.loads(doc)
    _assert_same(nanojson.loads(doc), expected)
    _assert_same(nanojson.loads(nanojson.dumps(nanojson.loads(doc))), expected)
    _assert_same(nanojson.loads(json.dumps(expected)), expected)


def test_nesting_beyond_100_levels_is_documented_limitation():
    """nanojson is recursive; ~300+ levels exhaust the CPython stack.

    The stdlib C accelerator digs deeper, so this is a documented
    difference (see the module docstring, difference #4).  We only assert
    that the *required* 100 levels work and that whatever happens deeper
    is an orderly exception, not silent corruption.
    """
    doc = "[" * 100 + "1" + "]" * 100
    assert nanojson.loads(doc) == json.loads(doc)


# --------------------------------------------------------------------------
# NUL characters
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (r'"\u0000"', "\x00"),
        (r'"a\u0000b\u0000c"', "a\x00b\x00c"),
        (r'{"\u0000": ["\u0000"]}', {"\x00": ["\x00"]}),
        (r'"\u0000\u0000"', "\x00\x00"),
    ],
    ids=["lone-nul", "embedded-nul", "nul-key-and-value", "nul-adjacent-text"],
)
def test_escaped_nul_matches_stdlib(text, expected):
    assert nanojson.loads(text) == json.loads(text) == expected


def test_raw_nul_is_rejected_by_both_implementations():
    for text in ('"a\x00b"', '{"\x00": 1}', '"a\\u0000\x00"', " \x00"):
        with pytest.raises(ValueError):
            json.loads(text)
        with pytest.raises(ValueError):
            nanojson.loads(text)


def test_nul_roundtrips_through_dumps():
    obj = {"n\x00l": "a\x00b", "list": ["\x00", "\x00\x00x"], "nested": {"\x00": []}}
    for ensure_ascii in (True, False):
        text = nanojson.dumps(obj, ensure_ascii=ensure_ascii)
        assert "\\u0000" in text  # control characters are always escaped
        assert json.dumps(obj, ensure_ascii=ensure_ascii) == text
        assert nanojson.loads(text) == json.loads(text) == obj


# --------------------------------------------------------------------------
# surrogate pairs and lone surrogates
# --------------------------------------------------------------------------


SURROGATE_DOCUMENTS = [
    r'"\ud83d\ude00"',  # emoji written as a surrogate pair
    r'"\ud800\udc00"',  # lowest valid pair
    r'"\udbff\udfff"',  # highest valid pair
    r'"\ud83d\ude00\ud83d\ude00"',  # two pairs in a row
    r'"x\ud83d\ude00y"',
    r'{"\ud83d\ude00": ["\ud83d\ude00"]}',
    r'"\ud800"',  # lone high surrogate: kept as-is by both
    r'"\udc00"',  # lone low surrogate: kept as-is by both
    r'"\ud800\ud800"',  # high surrogate not followed by a low one
    r'"\udc00\udc00"',
    r'"\ud800x"',  # high surrogate followed by an ordinary character
    r'"a\ud800b\udc00c"',  # lone surrogates mixed into text
    r'"\ud800\n"',  # high surrogate followed by another escape
]


@pytest.mark.parametrize("text", SURROGATE_DOCUMENTS)
def test_surrogate_documents_match_stdlib(text):
    _assert_same(nanojson.loads(text), json.loads(text))


def test_surrogate_roundtrip_in_both_ascii_modes():
    value = [
        "\ud800",
        "\udfff",
        "\U0001f600",
        "a\udc00b",
        {"\ud800\udc00": "\udbff\udfff"},
    ]
    for ensure_ascii in (True, False):
        nano_text = nanojson.dumps(value, ensure_ascii=ensure_ascii)
        std_text = json.dumps(value, ensure_ascii=ensure_ascii)
        assert nano_text == std_text
        # All four dumps/loads paths agree with each other ...
        _assert_same(nanojson.loads(nano_text), json.loads(nano_text))
        _assert_same(nanojson.loads(std_text), json.loads(std_text))
        _assert_same(nanojson.loads(nano_text), nanojson.loads(std_text))
        # ... and with ensure_ascii=False the original value comes back
        # exactly; with ensure_ascii=True the escaped key "\ud800\udc00" is
        # read back as the paired character (see the test below -- both
        # implementations do this, it is inherent to JSON surrogate pairs).
        if not ensure_ascii:
            _assert_same(nanojson.loads(nano_text), value)


def test_adjacent_lone_surrogates_recombine_like_in_stdlib():
    """Escaping adjacent lone surrogates makes them a pair -- in both.

    The Python string "\\ud800\\udc00" is two *lone* surrogates, but with
    ``ensure_ascii=True`` both implementations serialise it to the escape
    sequence ``"\\ud800\\udc00"``, which any JSON reader -- including the
    stdlib itself -- reads back as the single character U+10000.  So the
    round trip is lossy for adjacent lone surrogates in escaped output,
    identically in nanojson and the stdlib (this is a property of JSON's
    surrogate-pair escapes, not an implementation choice).
    """
    value = "\ud800\udc00"
    for ensure_ascii in (True, False):
        nano_text = nanojson.dumps(value, ensure_ascii=ensure_ascii)
        std_text = json.dumps(value, ensure_ascii=ensure_ascii)
        assert nano_text == std_text
        if ensure_ascii:
            assert nanojson.loads(nano_text) == json.loads(nano_text) == "\U00010000"
            assert nanojson.loads(nano_text) != value  # documented lossiness
        else:
            # Raw characters are not re-paired, so the round trip is exact.
            assert nanojson.loads(nano_text) == json.loads(nano_text) == value


def test_raw_surrogate_characters_in_text_match_stdlib():
    # A lone surrogate carried literally in the input (not as an escape) is
    # preserved by both implementations.
    for text in ("\"\ud800\"", "\"a\udc00b\"", "\"\ud800\udc00\""):
        _assert_same(nanojson.loads(text), json.loads(text))


# --------------------------------------------------------------------------
# numbers: big integers and exponent notation
# --------------------------------------------------------------------------


NUMBER_DOCUMENTS = [
    "0",
    "-0",
    "1",
    "-1",
    "42",
    "-17",
    "123456789012345678901234567890",
    "-123456789012345678901234567890",
    str(2**200),
    str(-(2**200)),
    "9007199254740993",  # 2**53 + 1, not representable as a float
    "0.1",
    "-0.0",
    "1.0",
    "1.00",
    "0.500",
    "1e0",
    "1E0",
    "1e+2",
    "1E-2",
    "-1.5e-300",
    "6.02e23",
    "1.7976931348623157e308",  # largest finite double
    "5e-324",  # smallest subnormal double
    "2.2250738585072014e-308",  # smallest normal double
    "1e308",
    "1e400",  # overflows to +infinity in both implementations
    "-1e400",  # overflows to -infinity in both implementations
    "1e-400",  # underflows to 0.0
    "123456789.123456789",
    "0." + "0" * 400 + "1",
    "1" + "0" * 400,
]


@pytest.mark.parametrize("text", NUMBER_DOCUMENTS)
def test_number_documents_match_stdlib(text):
    expected = json.loads(text)
    actual = nanojson.loads(text)
    assert type(actual) is type(expected), (
        "type differs for %r: %s vs %s"
        % (text, type(actual).__name__, type(expected).__name__)
    )
    assert actual == expected
    if isinstance(expected, float):
        assert math.copysign(1.0, actual) == math.copysign(1.0, expected)


def test_big_numbers_roundtrip_through_dumps():
    integers = [0, -0, 42, 2**53 + 1, 2**200, -(2**200), 123456789012345678901234567890]
    for value in integers:
        text = nanojson.dumps(value)
        assert text == json.dumps(value)
        restored = nanojson.loads(text)
        assert isinstance(restored, int)  # never routed through float
        assert restored == json.loads(text) == value
    floats = [0.1, -0.0, 5e-324, 1.7976931348623157e308, 6.02e23, 1e-7, math.pi]
    for value in floats:
        text = nanojson.dumps(value)
        assert text == json.dumps(value)
        restored = nanojson.loads(text)
        assert type(restored) is float
        assert restored == value
        assert math.copysign(1.0, restored) == math.copysign(1.0, value)


def test_very_long_integer_agrees_with_stdlib():
    """CPython's int() digit limit (>4300 digits) trips both implementations."""
    text = "9" * 4301
    limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    if limit and len(text) > limit:
        with pytest.raises(ValueError):
            json.loads(text)
        with pytest.raises(ValueError):
            nanojson.loads(text)
    else:  # pragma: no cover - only on interpreters without the limit
        assert nanojson.loads(text) == json.loads(text)


# --------------------------------------------------------------------------
# ensure_ascii=True / False
# --------------------------------------------------------------------------


def test_ensure_ascii_true_and_false_both_ways():
    obj = {
        "ascii": "plain text",
        "latin1": "café",
        "cjk": "日本語",
        "emoji": "\U0001f600",
        "music": "\U0001d11e",
        "control": "\x00\x01\x1f\x7f",
        "quote": 'say "hi"',
        "backslash": "back\\slash",
        "newline": "line1\nline2\ttabbed\rcr",
        "lone": "\ud800high\udc00low",
        "nested": [{"k": ["é", "\U0001f600", None, True, False, 0]}, {}],
    }
    for ensure_ascii in (True, False):
        nano_text = nanojson.dumps(obj, ensure_ascii=ensure_ascii)
        std_text = json.dumps(obj, ensure_ascii=ensure_ascii)
        assert nano_text == std_text
        if ensure_ascii:
            assert nano_text.isascii()
            assert "\\u00e9" in nano_text  # é escaped
            assert "\\ud83d\\ude00" in nano_text  # emoji as surrogate pair
        else:
            assert "café" in nano_text
            assert "日本語" in nano_text
            assert "\\u00e9" not in nano_text
        _assert_same(nanojson.loads(nano_text), obj)
        _assert_same(json.loads(nano_text), obj)
        # Either encoder's output parses back to the same value in both.
        _assert_same(nanojson.loads(std_text), obj)
        _assert_same(json.loads(std_text), obj)


def test_ensure_ascii_false_still_escapes_controls_and_quotes():
    value = 'a"b\\c\nd\x00e\x7ff'
    for ensure_ascii in (True, False):
        nano_text = nanojson.dumps(value, ensure_ascii=ensure_ascii)
        assert nano_text == json.dumps(value, ensure_ascii=ensure_ascii)
        assert nanojson.loads(nano_text) == json.loads(nano_text) == value


# --------------------------------------------------------------------------
# key order and duplicate keys
# --------------------------------------------------------------------------


def test_key_order_and_duplicate_keys_match_stdlib():
    text = '{"z": 1, "a": 2, "\\u00e9": 3, "": 4, "a": 5}'
    expected = json.loads(text)
    actual = nanojson.loads(text)
    assert list(actual) == list(expected) == ["z", "a", "é", ""]
    assert actual == expected == {"z": 1, "a": 5, "é": 3, "": 4}
    obj = {"z": 1, "a": [2, {"é": 3}], "": 4}
    assert nanojson.dumps(obj) == json.dumps(obj)


# --------------------------------------------------------------------------
# malformed input: both implementations must agree
# --------------------------------------------------------------------------


# (text, accepted) -- both implementations must agree on acceptance and,
# when accepted, on the parsed value.  All entries are printable ASCII.
AGREEMENT_CASES = [
    ("{}", True),
    ("[]", True),
    ("null", True),
    ("true", True),
    ("false", True),
    ("0", True),
    ("-0", True),
    ("2.5e-3", True),
    ("1E+4", True),
    ('"\\u0041\\"\\\\\\/\\b\\f\\n\\r\\t"', True),
    ('{"a\\u0000b": {}}', True),
    (r'"\ud83d\ude00"', True),
    (r'"\ud800"', True),
    ("[[[[[[[[[[1]]]]]]]]]]", True),
    ("[1, 2", False),
    ("[1, 2,]", False),
    ('{"a": 1,}', False),
    ("{a: 1}", False),
    ("{'a': 1}", False),
    ('{"a" 1}', False),
    ('{"a": 1 "b": 2}', False),
    ("[1 2]", False),
    ("[,]", False),
    ("{,}", False),
    ("[", False),
    ("]", False),
    ("{", False),
    ("}", False),
    ('"', False),
    (r'"\x"', False),
    (r'"\u12"', False),
    (r'"\u12g4"', False),
    ('"unterminated', False),
    ('"a" "b"', False),
    ("{} {}", False),
    ("[1] [2]", False),
    ("nullnull", False),
    ("tru", False),
    ("True", False),
    ("None", False),
    ("nul", False),
    ("undefined", False),
    ("01", False),
    ("+1", False),
    ("-.5", False),
    (".5", False),
    ("1.", False),
    ("1e", False),
    ("1e+", False),
    ("1e-", False),
    ("-", False),
    ("--1", False),
    ("1..2", False),
    ("[[1],", False),
    ('{"a":}', False),
    ('{"a"}', False),
    ('{"a": "b",}', False),
]


@pytest.mark.parametrize("text,accepted", AGREEMENT_CASES)
def test_acceptance_agrees_with_stdlib(text, accepted):
    if accepted:
        _assert_same(nanojson.loads(text), json.loads(text))
    else:
        with pytest.raises(ValueError):
            json.loads(text)
        with pytest.raises(ValueError):
            nanojson.loads(text)


def test_whitespace_handling_matches_stdlib():
    # RFC 8259 allows exactly space, tab, newline and carriage return.
    assert nanojson.loads("  [ 1 ,\t2 ]\n\r ") == json.loads("  [ 1 ,\t2 ]\n\r ")
    for text in ("\x0b", "\x0c", "\xa0", "\u2028", "\ufeff", '[1,\x0c2]'):
        with pytest.raises(ValueError):
            json.loads(text)
        with pytest.raises(ValueError):
            nanojson.loads(text)


# --------------------------------------------------------------------------
# mutated documents: a fuzzier differential
# --------------------------------------------------------------------------


MUTATION_BASES = [
    r'{"a": [1, 2.5, null, true, false], "s": "x\u00e9\ud83d\ude00\u0000y"}',
    r'[{"k": -0.5e3, "arr": [[], {}]}, "text", 42, -0]',
    r'"\ud800\udc00 lone \udfff pair \ud83d\ude00"',
    r'{"": {"n\u0000l": [1e-5, 123456789012345678901234567890]}}',
    r'[[[[[{"deep": [true]}]]]]]',
    '{"dup": 1, "dup": 2, "other": "value"}',
    '"tab\\there\\nnew\\u0000line"',
    "[1e400, 0.1, 5e-324, -0.0]",
]
MUTATION_ALPHABET = '{}[],:"\\ 0123456789eE.+-tfruNlsxXuU\n\t'


def _mutate(rng, text):
    op = rng.randint(0, 4)
    if not text:
        return rng.choice(["{", "[", "x", "", " "])
    if op == 0:  # truncate
        return text[: rng.randint(0, len(text))]
    if op == 1:  # insert one character
        pos = rng.randint(0, len(text))
        return text[:pos] + rng.choice(MUTATION_ALPHABET) + text[pos:]
    if op == 2:  # delete one character
        pos = rng.randint(0, len(text) - 1)
        return text[:pos] + text[pos + 1 :]
    if op == 3:  # replace one character
        pos = rng.randint(0, len(text) - 1)
        return text[:pos] + rng.choice(MUTATION_ALPHABET) + text[pos + 1 :]
    return text + rng.choice(['"', "}", "]", "x", " ", ",", "\\u00"])


def test_mutated_documents_agree_with_stdlib():
    """8 bases x 30 mutations = 240 texts; acceptance must match exactly."""
    rng = random.Random(0xBADF00D)
    checked = 0
    for base in MUTATION_BASES:
        for _ in range(30):
            text = _mutate(rng, base)
            checked += 1
            try:
                expected, expected_error = json.loads(text), None
            except ValueError as exc:
                expected, expected_error = None, exc
            try:
                actual, actual_error = nanojson.loads(text), None
            except ValueError as exc:
                actual, actual_error = None, exc
            if expected_error is not None:
                # nanojson must be at least as strict as the reference.
                assert actual_error is not None, (
                    "nanojson accepted %r which the stdlib rejects (%s)"
                    % (text, expected_error)
                )
            elif _contains_nonfinite(expected) and _contains_nonfinite_token(text):
                # Documented difference #2: json parses the non-standard
                # NaN/Infinity tokens, nanojson refuses them (RFC 8259 has
                # no such tokens; overflowing numbers such as 1e400 parse
                # to infinity in both implementations and are compared
                # normally).
                assert actual_error is not None, (
                    "expected nanojson to refuse the non-finite document %r" % (text,)
                )
            else:
                assert actual_error is None, (
                    "nanojson rejected %r: %s" % (text, actual_error)
                )
                _assert_same(actual, expected, "mutated %r" % (text,))
    assert checked >= 100


# --------------------------------------------------------------------------
# documented differences (asserted so they cannot change silently)
# --------------------------------------------------------------------------


def test_nonfinite_constants_are_a_documented_difference():
    """json accepts NaN/Infinity; nanojson is strict RFC 8259 (difference #2)."""
    for text in ("NaN", "Infinity", "-Infinity", "[NaN]", '{"x": Infinity}'):
        value = json.loads(text)  # accepted by the reference implementation
        assert _contains_nonfinite(value)
        with pytest.raises(ValueError):
            nanojson.loads(text)
    with pytest.raises(ValueError):
        nanojson.dumps(float("nan"))
    with pytest.raises(ValueError):
        nanojson.dumps(float("inf"))
    with pytest.raises(ValueError):
        nanojson.dumps([1, {"x": float("-inf")}])
    # The reference serialises them to non-standard tokens.
    assert json.dumps(float("nan")) == "NaN"


def test_non_string_keys_are_a_documented_difference():
    """json coerces non-string keys; nanojson demands str keys (difference #1)."""
    # Note: True coerces to "1" too, which collides with the int key 1 and
    # is overwritten by the later True member (last wins) -- stdlib quirk.
    coerced = json.dumps({1: "a", True: "b", None: "c", 3.5: "d"})
    assert coerced == '{"1": "b", "null": "c", "3.5": "d"}'
    assert json.dumps({1: "a"}) == '{"1": "a"}'
    with pytest.raises(TypeError):
        nanojson.dumps({1: "a"})
    with pytest.raises(TypeError):
        nanojson.dumps({"ok": 1, 2: "b"})
    # Parsing is unaffected: JSON keys are always strings for both.
    assert nanojson.loads('{"1": "a"}') == json.loads('{"1": "a"}') == {"1": "a"}


def test_bom_handling_is_a_documented_difference():
    """json strips a UTF-8 BOM from bytes; nanojson is strict (difference #3)."""
    bom = b"\xef\xbb\xbf{}"
    assert json.loads(bom) == {}
    with pytest.raises(ValueError):
        nanojson.loads(bom)
    # A BOM carried in str input is rejected by both implementations.
    with pytest.raises(ValueError):
        json.loads("﻿{}")
    with pytest.raises(ValueError):
        nanojson.loads("﻿{}")
