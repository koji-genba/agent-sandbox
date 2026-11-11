"""Tests for nanojson.parser (and nanojson.load)."""

import io

import pytest

from nanojson import JSONParseError, load, loads


# --------------------------------------------------------------- literals


def test_literal_true_false_null():
    assert loads("true") is True
    assert loads("false") is False
    assert loads("null") is None


def test_integers():
    assert loads("0") == 0
    assert loads("-0") == 0
    assert loads("42") == 42
    assert loads("-17") == -17
    assert isinstance(loads("42"), int)


def test_big_integer_keeps_precision():
    big = 123456789012345678901234567890
    assert loads("123456789012345678901234567890") == big
    assert isinstance(loads("123456789012345678901234567890"), int)


def test_floats():
    assert loads("0.5") == 0.5
    assert loads("-0.5") == -0.5
    assert loads("3.14159") == 3.14159
    assert isinstance(loads("0.5"), float)


def test_exponent_notation():
    assert loads("1e10") == 1e10
    assert loads("1E+2") == 100.0
    assert loads("-1.5e-3") == -0.0015
    assert loads("6.02e23") == 6.02e23
    assert loads("1e-6") == 1e-6
    assert isinstance(loads("1e10"), float)


def test_number_without_fraction_or_exponent_stays_int():
    assert isinstance(loads("-7"), int)


# --------------------------------------------------------------- strings


def test_string_simple():
    assert loads('"hello"') == "hello"
    assert loads('""') == ""


def test_named_escapes():
    text = r'"\"\\\/\b\f\n\r\t"'
    expected = '"' + "\\" + "/" + "\b" + "\f" + "\n" + "\r" + "\t"
    assert loads(text) == expected


def test_unicode_escape():
    assert loads(r'"\u0041"') == "A"
    assert loads(r'"\u00e9"') == "é"
    assert loads(r'"\u4e2d"') == "中"
    assert loads(r'"\u0000"') == "\x00"


def test_surrogate_pair():
    assert loads(r'"\ud83d\ude00"') == "😀"
    assert loads(r'"\ud83d\ude00"') == "\U0001F600"


def test_lone_surrogate_is_preserved():
    # RFC 8259 permits this grammatically; the behaviour is ours to define.
    assert loads(r'"\ud800"') == "\ud800"
    assert loads(r'"\udc00"') == "\udc00"


def test_high_surrogate_not_followed_by_low_surrogate():
    assert loads(r'"\ud800\ud800"') == "\ud800\ud800"
    assert loads(r'"\ud800A"') == "\ud800A"


def test_raw_non_ascii_characters():
    assert loads('"日本語🌍"') == "日本語🌍"


def test_invalid_escape_raises():
    with pytest.raises(JSONParseError):
        loads(r'"\x"')


def test_truncated_unicode_escape_raises():
    with pytest.raises(JSONParseError):
        loads(r'"\u12"')


def test_non_hex_unicode_escape_raises():
    with pytest.raises(JSONParseError):
        loads(r'"\u12g4"')


def test_unterminated_string_raises():
    with pytest.raises(JSONParseError):
        loads('"abc')


def test_trailing_backslash_raises():
    with pytest.raises(JSONParseError):
        loads('"abc\\')


def test_raw_control_character_in_string_raises():
    with pytest.raises(JSONParseError):
        loads('"a\nb"')
    with pytest.raises(JSONParseError):
        loads('"a\tb"')


# --------------------------------------------------------------- objects


def test_object_basic():
    assert loads('{"a": 1}') == {"a": 1}
    assert loads('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_object_empty():
    assert loads("{}") == {}


def test_object_nested():
    doc = '{"a": {"b": {"c": [1, 2, {"d": null}]}}}'
    assert loads(doc) == {"a": {"b": {"c": [1, 2, {"d": None}]}}}


def test_object_key_order_preserved():
    assert list(loads('{"b": 1, "a": 2}')) == ["b", "a"]


def test_duplicate_keys_last_wins():
    assert loads('{"a": 1, "a": 2}') == {"a": 2}


def test_object_keys_may_be_empty_strings():
    assert loads('{"": 0}') == {"": 0}


def test_unclosed_object_raises():
    with pytest.raises(JSONParseError):
        loads('{"a": 1')


def test_unclosed_nested_structure_raises():
    with pytest.raises(JSONParseError):
        loads('{"a": [1, {"b": ')


def test_missing_colon_raises():
    with pytest.raises(JSONParseError):
        loads('{"a" 1}')


def test_missing_comma_between_members_raises():
    with pytest.raises(JSONParseError):
        loads('{"a": 1 "b": 2}')


def test_trailing_comma_in_object_raises():
    with pytest.raises(JSONParseError):
        loads('{"a": 1,}')


def test_unquoted_key_raises():
    with pytest.raises(JSONParseError):
        loads("{a: 1}")


def test_single_quoted_key_raises():
    with pytest.raises(JSONParseError):
        loads("{'a': 1}")


# --------------------------------------------------------------- arrays


def test_array_basic():
    assert loads("[1, 2, 3]") == [1, 2, 3]
    assert loads("[1, \"two\", null, true]") == [1, "two", None, True]


def test_array_empty():
    assert loads("[]") == []


def test_array_nested():
    assert loads("[[1, 2], [3]]") == [[1, 2], [3]]
    assert loads('[{"a": []}]') == [{"a": []}]


def test_deeply_nested_arrays():
    depth = 100
    doc = "[" * depth + "1" + "]" * depth
    expected = 1
    for _ in range(depth):
        expected = [expected]
    assert loads(doc) == expected


def test_unclosed_array_raises():
    with pytest.raises(JSONParseError):
        loads("[1, 2")


def test_missing_comma_between_items_raises():
    with pytest.raises(JSONParseError):
        loads("[1 2]")


def test_trailing_comma_in_array_raises():
    with pytest.raises(JSONParseError):
        loads("[1,]")


def test_comma_only_array_raises():
    with pytest.raises(JSONParseError):
        loads("[,]")


# --------------------------------------------------------------- whitespace


def test_insignificant_whitespace_everywhere():
    doc = '  \t\n\r {  "a" :\t[ 1 ,\n 2 ] \r }\n '
    assert loads(doc) == {"a": [1, 2]}


def test_escaped_whitespace_inside_string_is_significant():
    assert loads(r'"a b\tc"') == "a b" + "\t" + "c"
    assert loads(r'"line\nbreak"') == "line" + "\n" + "break"


# --------------------------------------------------------------- top level


def test_empty_input_raises():
    with pytest.raises(JSONParseError):
        loads("")


def test_whitespace_only_input_raises():
    with pytest.raises(JSONParseError):
        loads("  \n\t ")


def test_invalid_literal_tokens_raise():
    for bad in ("tru", "True", "none", "undefined", "NaN", "Infinity", "-Infinity"):
        with pytest.raises(JSONParseError):
            loads(bad)


def test_bare_word_raises():
    with pytest.raises(JSONParseError):
        loads("abc")


def test_extra_data_after_value_raises():
    for bad in ("{} {}", "1 2", "true false", "[] []", '"a" "b"', "1,"):
        with pytest.raises(JSONParseError):
            loads(bad)


def test_garbage_before_value_raises():
    with pytest.raises(JSONParseError):
        loads("x {}")


def test_byte_order_mark_is_rejected():
    with pytest.raises(JSONParseError):
        loads("\ufeff{}")


def test_invalid_number_tokens_raise():
    for bad in ("+1", "01", "00", "-01", "1.", ".5", "1e", "1e+", "1e-", "1.e5", "-"):
        with pytest.raises(JSONParseError):
            loads(bad)


def test_parse_error_is_valueerror_with_position():
    with pytest.raises(ValueError):
        loads("{")
    with pytest.raises(JSONParseError) as excinfo:
        loads("{")
    err = excinfo.value
    assert err.lineno == 1
    assert err.colno == 2
    assert err.pos == 1
    assert "line 1" in str(err)


def test_parse_error_position_on_first_line():
    with pytest.raises(JSONParseError) as excinfo:
        loads('{"a": tru}')
    assert excinfo.value.lineno == 1
    assert excinfo.value.colno == 7
    assert excinfo.value.pos == 6


def test_parse_error_position_on_later_line():
    doc = '{\n  "a": 1,\n  "b": 2\n'
    with pytest.raises(JSONParseError) as excinfo:
        loads(doc)
    assert excinfo.value.lineno == 4
    assert excinfo.value.colno == 1


def test_error_message_mentions_position():
    with pytest.raises(JSONParseError) as excinfo:
        loads("[1, 2")
    assert "line 1 column 6 (char 5)" in str(excinfo.value)


# --------------------------------------------------------------- load()


def test_load_from_stringio():
    assert load(io.StringIO('{"a": [1, 2]}')) == {"a": [1, 2]}


def test_load_from_text_file(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"日本語": [true, null], "n": -1.5e2}', encoding="utf-8")
    with open(path, encoding="utf-8") as fp:
        assert load(fp) == {"日本語": [True, None], "n": -150.0}


def test_load_propagates_parse_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[1, 2", encoding="utf-8")
    with open(path, encoding="utf-8") as fp:
        with pytest.raises(JSONParseError):
            load(fp)


def test_loads_accepts_bytes():
    assert loads(b'"abc"') == "abc"
    assert loads('{"a": 1}'.encode("utf-8")) == {"a": 1}


def test_loads_rejects_other_types():
    with pytest.raises(TypeError):
        loads(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        loads(None)  # type: ignore[arg-type]
