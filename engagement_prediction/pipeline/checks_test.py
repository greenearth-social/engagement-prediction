import json
import logging

import pytest

from engagement_prediction.experiment_tracking import NoOpExperimentTracker
from engagement_prediction.pipeline.checks import report_checks


class RecordingTracker(NoOpExperimentTracker):
    id = "checks-test-task"

    def __init__(self, upload_error):
        self.upload_error = upload_error
        self.reports = []
        self.failure_reason = None

    def log_file_artifact(self, name, path):
        self.reports.append((name, json.loads(path.read_text())))
        if self.upload_error:
            raise RuntimeError("upload unavailable")
        return True

    def mark_failed(self, reason):
        assert self.reports, "Save and attempt to upload diagnostics before changing task status"
        self.failure_reason = reason


@pytest.mark.parametrize("upload_error", [False, True])
def test_failure_is_saved_and_reported_before_raising(tmp_path, caplog, upload_error):
    report = {"status": "fail", "checks": [{"status": "fail", "message": "likes missing on 2026-01-02"}]}
    path = tmp_path / "run" / "raw_data_checks.json"
    tracker = RecordingTracker(upload_error)

    with pytest.raises(ValueError, match="likes missing on 2026-01-02"):
        report_checks(report, path=path, tracker=tracker, logger=logging.getLogger("checks_test"))

    assert json.loads(path.read_text()) == report
    assert tracker.reports == [("raw_data_checks", report)]
    assert "likes missing on 2026-01-02" in tracker.failure_reason
    assert "likes missing on 2026-01-02" in caplog.text
    if upload_error:
        assert "upload unavailable" in caplog.text


def test_warning_is_recorded_without_stopping_run(tmp_path, caplog):
    report = {"status": "warn", "checks": [{"status": "warn", "message": "low daily replies"}]}
    tracker = RecordingTracker(False)

    report_checks(
        report, path=tmp_path / "checks.json", tracker=tracker,
        logger=logging.getLogger("checks_test"),
    )

    assert tracker.reports == [("checks", report)]
    assert "low daily replies" in caplog.text
    assert tracker.failure_reason is None


def test_failed_upload_return_value_warns_without_masking_data_failure(tmp_path, caplog):
    class FailedUploadTracker(NoOpExperimentTracker):
        id = "failed-upload-task"

    report = {"status": "fail", "checks": [{"status": "fail", "message": "no likes"}]}
    with pytest.raises(ValueError, match="no likes"):
        report_checks(
            report, path=tmp_path / "checks.json", tracker=FailedUploadTracker(),
            logger=logging.getLogger("checks_test"),
        )

    assert "Could not upload check report" in caplog.text
    assert json.loads((tmp_path / "checks.json").read_text()) == report


def test_failed_status_update_does_not_hide_check_failure(tmp_path, caplog):
    class UnavailableTracker(NoOpExperimentTracker):
        def mark_failed(self, reason):
            raise RuntimeError("status update unavailable")

    report = {"status": "fail", "checks": [{"status": "fail", "message": "no replies"}]}
    with pytest.raises(ValueError, match="no replies"):
        report_checks(
            report, path=tmp_path / "checks.json", tracker=UnavailableTracker(),
            logger=logging.getLogger("checks_test"),
        )
    assert "status update unavailable" in caplog.text
