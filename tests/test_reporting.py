from pathlib import Path

from src.utils.reporting import generate_report


def test_generate_report_includes_fold_subject_metadata(tmp_path):
    report_path = generate_report(
        experiment_name="metadata_check",
        config={},
        fold_results=[
            {
                "weighted_f1": 0.5,
                "ccc": 0.2,
                "test_subject": "001",
                "val_subject": "002",
                "train_subjects": ["003", "004"],
                "n_train": 10,
                "n_val": 3,
                "n_test": 4,
            }
        ],
        report_dir=str(tmp_path),
    )

    report = Path(report_path).read_text()
    assert "## Per-Fold Results" in report
    assert "| 001 | 002 | 003,004 | 10 | 3 | 4 |" in report
