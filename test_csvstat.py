"""Tests for csvstat.py (pytest, standard library only).

Run with:  python3 -m pytest test_csvstat.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import csvstat


# ---------------------------------------------------------------- helpers

def run_main(tmp_path: Path, content: str, filename: str = "data.csv"):
    """Write *content* to a temp CSV, run csvstat.main on it, capture output."""
    csv_file = tmp_path / filename
    csv_file.write_text(content, encoding="utf-8")
    rc = csvstat.main([str(csv_file)])
    return rc, csv_file


# ---------------------------------------------------------------- 正常系

def test_normal_csv(tmp_path, capsys):
    """正常系: 3列のCSVの各列について集計が正しいこと。"""
    content = (
        "name,age,city\n"
        "alice,30,Tokyo\n"
        "bob,25,Osaka\n"
        "carol,35,Tokyo\n"
    )
    rc, _ = run_main(tmp_path, content)
    out = capsys.readouterr().out

    assert rc == 0
    lines = out.splitlines()
    assert len(lines) == 3
    assert lines[0] == "name: non-empty=3, missing=0, unique=3"
    assert lines[1] == "age: non-empty=3, missing=0, unique=3"
    # city は Tokyo が2回出るので一意値は2
    assert lines[2] == "city: non-empty=3, missing=0, unique=2"


def test_header_only(tmp_path, capsys):
    """正常系: ヘッダのみ（データ行なし）なら全カウント0。"""
    rc, _ = run_main(tmp_path, "a,b,c\n")
    out = capsys.readouterr().out

    assert rc == 0
    assert out.splitlines() == [
        "a: non-empty=0, missing=0, unique=0",
        "b: non-empty=0, missing=0, unique=0",
        "c: non-empty=0, missing=0, unique=0",
    ]


def test_quoted_cells(tmp_path, capsys):
    """正常系: csv モジュール経由で引用符・カンマ入りセルも正しく集計。"""
    content = 'name,tags\n"alice","a,b"\n"bob","a,b"\n'
    rc, _ = run_main(tmp_path, content)
    out = capsys.readouterr().out

    assert rc == 0
    assert out.splitlines() == [
        "name: non-empty=2, missing=0, unique=2",
        "tags: non-empty=2, missing=0, unique=1",
    ]


# ---------------------------------------------------------------- 空ファイル

def test_empty_file(tmp_path, capsys):
    """空ファイル: 列が存在しないため何も表示せず正常終了すること。"""
    rc, _ = run_main(tmp_path, "")
    out = capsys.readouterr().out

    assert rc == 0
    assert out == ""


def test_blank_lines_only(tmp_path, capsys):
    """空ファイル相当: 改行・空行のみのファイルも列なしとして正常終了。"""
    rc, _ = run_main(tmp_path, "\n\n")
    out = capsys.readouterr().out

    assert rc == 0
    assert out == ""


# ---------------------------------------------------------------- 欠損あり

def test_missing_values(tmp_path, capsys):
    """欠損あり: 空セルと行末の欠けが missing として数えられること。"""
    content = (
        "name,age,city\n"
        "alice,30,Tokyo\n"
        ",25,\n"          # 空文字セル
        "carol,35\n"      # 行が短い（city のセルが欠落）
    )
    rc, _ = run_main(tmp_path, content)
    out = capsys.readouterr().out

    assert rc == 0
    lines = out.splitlines()
    # name: 2件あり / 1件欠損 / 一意値は alice, carol の2
    assert lines[0] == "name: non-empty=2, missing=1, unique=2"
    # age: 3件すべてあり（30/25/35 で一意値は3）
    assert lines[1] == "age: non-empty=3, missing=0, unique=3"
    # city: 1件あり / 2件欠損 / 一意値は Tokyo のみ
    assert lines[2] == "city: non-empty=1, missing=2, unique=1"


def test_whitespace_only_cell_counts_as_missing(tmp_path, capsys):
    """欠損あり: 空白のみのセルは欠損扱いで、一意値にも数えないこと。"""
    content = (
        "a,b\n"
        "x,   \n"
        "x,y\n"
    )
    rc, _ = run_main(tmp_path, content)
    out = capsys.readouterr().out

    assert rc == 0
    assert out.splitlines() == [
        "a: non-empty=2, missing=0, unique=1",
        "b: non-empty=1, missing=1, unique=1",
    ]


# ---------------------------------------------------------------- エラー系

def test_file_not_found(tmp_path, capsys):
    """エラー系: 存在しないファイルは stderr にメッセージを出し終了コード1。"""
    rc = csvstat.main([str(tmp_path / "nope.csv")])
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out == ""
    assert "error" in captured.err
    assert "nope.csv" in captured.err


def test_cli_subprocess(tmp_path):
    """エンドツーエンド: python3 csvstat.py FILE として起動できること。"""
    csv_file = tmp_path / "cli.csv"
    csv_file.write_text("id,val\n1,a\n2,\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(Path(csvstat.__file__).parent / "csvstat.py"),
         str(csv_file)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "id: non-empty=2, missing=0, unique=2",
        "val: non-empty=1, missing=1, unique=1",
    ]


def test_cli_no_args():
    """エラー系: 引数なしは使用法エラー（終了コード2）。"""
    with pytest.raises(SystemExit) as excinfo:
        csvstat.main([])
    assert excinfo.value.code == 2
