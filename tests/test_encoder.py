"""Tests for nanojson.encoder (nanojson.dumps)."""

import math

import pytest

from nanojson import dumps, loads


# --------------------------------------------------------------- scalars


def test_none():
    assert dumps(None) == "null"


def test_booleans():
    assert dumps(True) == "true"
    assert dumps(False) == "false"


def test_integers():
    assert dumps(0) == "0"
    assert dumps(42) == "42"
    assert dumps(-17) == "-17"
    assert dumps(123456789012345678901234567890) == "123456789012345678901234567890"


def test_floats():
    assert dumps(0.0) == "0.0"
    assert dumps(-0.0) == "-0.0"
    assert dumps(1.5) == "1.5"
    assert dumps(-0.25) == "-0.25"
    assert dumps(0.1) == "0.1"


def test_floats_use_exponent_notation():
    assert dumps(1e30) == "1e+30"
    assert dumps(-1.5e-7) == "-1.5e-07"
    assert dumps(1e16) == "1e+16"


def test_strings_ascii():
    assert dumps("hello") == '"hello"'
    assert dumps("") == '""'


def test_string_escapes():
    obj = '"' + "\\" + "/" + "\b" + "\f" + "\n" + "\r" + "\t"
    assert dumps(obj) == r'"\"\\/\b\f\n\r\t"'


def test_string_control_characters_escaped():
    assert dumps("\x00\x01\x1f") == r'"\u0000\u0001\u001f"'
    assert dumps("a\x0bb") == r'"a\u000bb"'


def test_ensure_ascii_escapes_non_ascii():
    assert dumps("é") == r'"\u00e9"'
    assert dumps("中") == r'"\u4e2d"'
    assert dumps("aé", ensure_ascii=True) == r'"a\u00e9"'


def test_ensure_ascii_escapes_non_bmp_as_surrogate_pair():
    assert dumps("😀") == r'"\ud83d\ude00"'
    assert dumps("𝄞") == r'"\ud834\udd1e"'


def test_ensure_ascii_false_keeps_unicode_literal():
    assert dumps("é中😀", ensure_ascii=False) == '"é中😀"'


def test_ensure_ascii_false_still_escapes_quotes_and_controls():
    assert dumps('a"b\\c\nd', ensure_ascii=False) == r'"a\"b\\c\nd"'


def test_del_character_is_escaped():
    assert dumps("\x7f") == r'"\u007f"'


# --------------------------------------------------------------- containers


def test_list():
    assert dumps([1, "two", None, True]) == '[1, "two", null, true]'


def test_tuple_is_encoded_as_array():
    assert dumps((1, 2, 3)) == "[1, 2, 3]"
    assert dumps(()) == "[]"


def test_nested_containers():
    assert dumps({"a": [1, {"b": (2, 3)}]}) == '{"a": [1, {"b": [2, 3]}]}'


def test_empty_containers():
    assert dumps([]) == "[]"
    assert dumps({}) == "{}"
    assert dumps([[], {}, [1]]) == "[[], {}, [1]]"


def test_dict_key_order_preserved():
    assert dumps({"b": 1, "a": 2}) == '{"b": 1, "a": 2}'


def test_dict_separators_default():
    assert dumps({"a": 1, "b": 2}) == '{"a": 1, "b": 2}'


def test_int_and_str_subclasses():
    class MyInt(int):
        pass

    class MyStr(str):
        pass

    assert dumps(MyInt(7)) == "7"
    assert dumps(MyStr("x")) == '"x"'
    assert dumps([MyInt(1), MyStr("y")]) == '[1, "y"]'


# --------------------------------------------------------------- indent


def test_indent_two():
    expected = '{\n  "a": [\n    1,\n    2\n  ],\n  "b": {\n    "c": true\n  }\n}'
    assert dumps({"a": [1, 2], "b": {"c": True}}, indent=2) == expected


def test_indent_four_nested_arrays():
    expected = "[\n    1,\n    [\n        2,\n        3\n    ]\n]"
    assert dumps([1, [2, 3]], indent=4) == expected


def test_indent_keeps_empty_containers_inline():
    expected = '{\n  "a": [],\n  "b": {}\n}'
    assert dumps({"a": [], "b": {}}, indent=2) == expected


def test_indent_string_is_used_verbatim():
    expected = "[\n\t1,\n\t[\n\t\t2\n\t]\n]"
    assert dumps([1, [2]], indent="\t") == expected


def test_indent_zero_still_adds_newlines():
    assert dumps([1], indent=0) == "[\n1\n]"


def test_indent_none_is_compact():
    assert dumps([1, [2]], indent=None) == "[1, [2]]"


def test_indent_negative_int_raises_typeerror():
    with pytest.raises(TypeError):
        dumps([1], indent=-2)


def test_indent_invalid_type_raises_typeerror():
    with pytest.raises(TypeError):
        dumps([1], indent=1.5)  # type: ignore[arg-type]


# --------------------------------------------------------------- errors


def test_non_string_dict_keys_raise_typeerror():
    for bad in ({1: "a"}, {None: 1}, {(1, 2): 1}, {True: 1}, {3.5: "x"}):
        with pytest.raises(TypeError):
            dumps(bad)


def test_nan_raises_valueerror():
    with pytest.raises(ValueError):
        dumps(float("nan"))


def test_infinity_raises_valueerror():
    with pytest.raises(ValueError):
        dumps(float("inf"))
    with pytest.raises(ValueError):
        dumps(float("-inf"))


def test_non_finite_float_nested_raises_valueerror():
    with pytest.raises(ValueError):
        dumps([1, {"a": [float("nan")]}])
    with pytest.raises(ValueError):
        dumps({"x": float("inf")})


def test_unsupported_types_raise_typeerror():
    for bad in ({1, 2}, frozenset([1]), object(), b"bytes", 1 + 2j, Ellipsis):
        with pytest.raises(TypeError):
            dumps(bad)


def test_circular_list_raises_valueerror():
    value = [1, 2]
    value.append(value)
    with pytest.raises(ValueError):
        dumps(value)


def test_circular_dict_raises_valueerror():
    value = {"a": 1}
    value["self"] = value
    with pytest.raises(ValueError):
        dumps(value)


def test_circular_nested_structure_raises_valueerror():
    inner = []
    outer = {"items": inner}
    inner.append(outer)
    with pytest.raises(ValueError):
        dumps(outer)


def test_shared_but_not_circular_object_is_fine():
    shared = [1, 2]
    assert dumps([shared, shared]) == "[[1, 2], [1, 2]]"
    shared_dict = {"k": 1}
    assert dumps({"a": shared_dict, "b": shared_dict}) == '{"a": {"k": 1}, "b": {"k": 1}}'


# --------------------------------------------------------------- round trip


def test_roundtrip_nested_structure():
    obj = {
        "int": 42,
        "neg": -17,
        "big": 123456789012345678901234567890,
        "float": 0.1,
        "exp": 6.02e23,
        "str": 'quote " backslash \\ slash / end\b\f\n\r\t',
        "unicode": "é中😀",
        "none": None,
        "list": [1, 2.5, "three", None, True, False],
        "nested_list": [1, [2, [3]]],
        "dict": {"nested": {"deep": []}},
    }
    assert loads(dumps(obj)) == obj


def test_roundtrip_tuples_decode_as_lists():
    assert loads(dumps((1, (2, [3])))) == [1, [2, [3]]]


def test_roundtrip_preserves_float_values():
    for value in (0.1, -0.25, 1e30, 6.02e23, 1e-7, 3.141592653589793):
        restored = loads(dumps(value))
        assert restored == value
        assert isinstance(restored, float)


def test_roundtrip_preserves_negative_zero():
    restored = loads(dumps(-0.0))
    assert restored == 0.0
    assert math.copysign(1.0, restored) == -1.0


def test_roundtrip_preserves_big_integers():
    big = 2 ** 200 + 12345
    restored = loads(dumps(big))
    assert restored == big
    assert isinstance(restored, int)


def test_roundtrip_special_strings():
    for text in (
        "",
        "plain",
        'quote " inside',
        "backslash \\ inside",
        "control \x00\x01\x1f\x7f",
        "unicode é 中 😀 𝄞",
        "line\nbreak\ttab\rcarriage",
    ):
        assert loads(dumps(text)) == text
        assert loads(dumps(text, ensure_ascii=False)) == text


def test_roundtrip_ensure_ascii_both_ways():
    obj = {"key": ["é中😀", {"nested": [1, 2.5, None]}]}
    for kwargs in ({"ensure_ascii": True}, {"ensure_ascii": False}):
        assert loads(dumps(obj, **kwargs)) == obj


def test_roundtrip_indented_output_is_reparsable():
    obj = {"a": [1, 2, {"b": None}], "c": [True, "x"]}
    text = dumps(obj, indent=3)
    assert "\n" in text
    assert loads(text) == obj


def test_roundtrip_parse_encode_parse_stable():
    doc = '{"n": [1, 2.5, null, true, false], "s": "a\\u00e9b\\ud83d\\ude00", "e": {}}'
    once = loads(doc)
    twice = loads(dumps(once))
    assert once == twice
    assert dumps(twice) == dumps(once)


def test_roundtrip_empty_and_nested_containers():
    for obj in ([], {}, [[]], {"a": {}}, [[[[]]]], {"a": [[{"b": []}]]}):
        assert loads(dumps(obj)) == obj
        assert loads(dumps(obj, indent=2)) == obj


def test_roundtrip_ascii_escaped_output_is_ascii_only():
    text = dumps({"msg": "é中😀\n"})
    assert text.isascii()
    assert loads(text) == {"msg": "é中😀\n"}
