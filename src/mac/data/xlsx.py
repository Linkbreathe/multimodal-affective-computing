from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = {"m": MAIN_NS}
CELL_RE = re.compile(r"([A-Z]+)(\d+)")


def _column_index(reference: str) -> int:
    letters = CELL_RE.fullmatch(reference).group(1)
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - 64
    return result - 1


def read_first_sheet(path: Path) -> list[list[str | float | None]]:
    """Dependency-free reader for the value table in a standard .xlsx workbook."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", NS):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")))
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        output: list[list[str | float | None]] = []
        for row in root.findall(".//m:sheetData/m:row", NS):
            values: dict[int, str | float | None] = {}
            for cell in row.findall("m:c", NS):
                index = _column_index(cell.attrib["r"])
                cell_type = cell.get("t")
                node = cell.find("m:v", NS)
                raw = None if node is None else node.text
                if cell_type == "s" and raw is not None:
                    value: str | float | None = shared[int(raw)]
                elif cell_type == "inlineStr":
                    value = "".join(t.text or "" for t in cell.iter(f"{{{MAIN_NS}}}t"))
                elif raw is None:
                    value = None
                else:
                    try:
                        value = float(raw)
                    except ValueError:
                        value = raw
                values[index] = value
            width = max(values, default=-1) + 1
            output.append([values.get(i) for i in range(width)])
        return output

