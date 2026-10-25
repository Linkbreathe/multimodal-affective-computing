"""Check curated real-data reports against frozen CSVs, without reanalysis."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/supplementary/participant_multimodal_20261025"
REPORTS = ROOT / "docs/supplementary/participant_multimodal_20261025"
RENDERER = runpy.run_path(str(ROOT / "scripts/summarize_participant_exploration.py"))


def test_profiles_match_frozen_tables():
    rendered = RENDERER["render"](SOURCE)
    assert rendered == (REPORTS / "participant_profiles_zh.txt").read_text(encoding="utf-8")
    assert rendered.count("===== P") == 9
    assert rendered.count("配对条件数=") == 198
    assert rendered.count("不可估计；原因=") == 39


def test_case_correlations_match_all_frozen_results():
    read_rows = RENDERER["read_rows"]
    variables = [
        r["variable"] for r in read_rows(SOURCE / "variables.csv") if r["modality"] != "Rating"
    ]
    lookup = {
        (r["participant_id"], r["feature"], r["outcome"]): r
        for r in read_rows(SOURCE / "individual_associations.csv")
    }
    text = (REPORTS / "P007_P009_comparison_zh.txt").read_text(encoding="utf-8")
    section = text.split("2. 完整个体内相关", 1)[1].split("ρ为Spearman", 1)[0]
    rows = [line.split(" | ")[1:] for line in section.splitlines() if " | " in line][1:]
    assert len(rows) == len(variables) == 11
    combinations = [
        ("P007", "relaxation"),
        ("P009", "relaxation"),
        ("P007", "discomfort"),
        ("P009", "discomfort"),
    ]
    for feature, values in zip(variables, rows, strict=True):
        for (participant, outcome), value in zip(combinations, values, strict=True):
            reference = lookup[participant, feature, outcome]["spearman_rho"]
            if not reference:
                assert value == "不可估计"
            else:
                assert float(value) == round(float(reference), 3)


def test_case_ratings_match_frozen_conditions():
    rows = RENDERER["read_rows"](SOURCE / "participant_condition_features.csv")
    lookup = {(r["participant_id"], r["condition"]): r for r in rows}
    text = (REPORTS / "P007_P009_comparison_zh.txt").read_text(encoding="utf-8")
    cases = [
        line.split(" | ") for line in text.splitlines() if line.startswith("C") and " | " in line
    ]
    assert len(cases) == 9
    for condition, first, second in cases:
        for participant, value in [("P007", first), ("P009", second)]:
            row = lookup[participant, condition]
            assert value == f"{row['relaxation_raw']}/{row['discomfort_raw']}"
