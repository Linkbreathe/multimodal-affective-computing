from __future__ import annotations

from pathlib import Path

from real_time_ml.data.io import iter_csv, parse_float, sniff_csv


def test_sep_comma(tmp_path: Path):
    path = tmp_path / "comma.csv"
    path.write_text("sep=,\na,b\n1,2\n", encoding="utf-8")
    assert sniff_csv(path) == (",", ".", 1)
    assert list(iter_csv(path))[0] == {"a": "1", "b": "2"}


def test_semicolon_decimal_comma_and_repeated_decimal_group(tmp_path: Path):
    path = tmp_path / "semi.csv"
    path.write_text("time;value\n1,78057E+12;33.615.214\n", encoding="utf-8")
    delimiter, decimal, skip = sniff_csv(path)
    assert (delimiter, decimal, skip) == (";", ",", 0)
    row = list(iter_csv(path))[0]
    assert parse_float(row["time"], decimal) == 1.78057e12
    assert abs(parse_float(row["value"], decimal) - 33.615214) < 1e-9

