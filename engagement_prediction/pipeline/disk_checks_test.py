from types import SimpleNamespace

import pytest

from engagement_prediction.pipeline import disk_checks


@pytest.mark.parametrize(
    ("output_free_bytes", "expected_status"),
    [(150 * 1024**3, "pass"), (150 * 1024**3 - 1, "fail")],
)
def test_output_free_space_threshold_includes_exact_boundary(
    monkeypatch, tmp_path, output_free_bytes, expected_status
):
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        disk_checks.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=output_free_bytes),
    )

    report = disk_checks.check_disk_space(
        output_dir=output_dir,
        min_output_free_gib=150,
    )

    assert report["status"] == expected_status
    [output] = report["checks"]
    assert output["status"] == expected_status
    assert output["free_bytes"] == output_free_bytes
    assert output["required_free_bytes"] == 150 * 1024**3
    assert output["path"] == str(output_dir)


def test_only_output_filesystem_is_checked(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"
    monkeypatch.setenv("TMPDIR", str(tmp_path / "temp"))
    checked_paths = []

    def disk_usage(path):
        checked_paths.append(path)
        return SimpleNamespace(free=200 * 1024**3 if path == output_dir else 0)

    monkeypatch.setattr(disk_checks.shutil, "disk_usage", disk_usage)

    report = disk_checks.check_disk_space(
        output_dir=output_dir,
        min_output_free_gib=150,
    )

    assert report["status"] == "pass"
    assert checked_paths == [output_dir]


def test_disk_usage_error_is_a_reported_failure(monkeypatch, tmp_path):
    output_dir = tmp_path / "output"

    def disk_usage(path):
        raise OSError("storage unavailable")

    monkeypatch.setattr(disk_checks.shutil, "disk_usage", disk_usage)

    report = disk_checks.check_disk_space(
        output_dir=output_dir,
        min_output_free_gib=150,
    )

    assert report["status"] == "fail"
    assert report["checks"][0]["free_bytes"] is None
    assert "storage unavailable" in report["checks"][0]["message"]
