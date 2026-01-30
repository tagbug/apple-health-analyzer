"""Coverage for CLI core helpers and guards."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src import cli as cli_module
from src.core.exceptions import HealthAnalyzerError


def test_report_to_dict_handles_none():
  """Ensure None report returns None."""
  assert cli_module._report_to_dict(None) is None


def test_report_to_dict_handles_missing_fields():
  """Ensure missing fields are handled safely."""
  payload = cli_module._report_to_dict(SimpleNamespace())

  assert payload["analysis_date"] is None
  assert payload["data_range"] is None
  assert payload["record_count"] == 0
  assert payload["data_quality_score"] == 0.0


def test_display_helpers_skip_optional_fields(monkeypatch):
  """Ensure display helpers handle optional fields gracefully."""
  console = MagicMock()
  console.print = MagicMock()
  monkeypatch.setattr(cli_module, "console", console)

  heart_rate_report = SimpleNamespace(
    resting_hr_analysis=None,
    hrv_analysis=None,
    cardio_fitness=None,
    data_quality_score=0.6,
    record_count=2,
  )
  sleep_report = SimpleNamespace(
    quality_metrics=None,
    data_quality_score=0.7,
    record_count=1,
  )
  highlights = SimpleNamespace(insights=[], recommendations=[])

  cli_module._display_heart_rate_results(heart_rate_report)
  cli_module._display_sleep_results(sleep_report)
  cli_module._display_highlights(highlights)

  assert console.print.call_count > 0


def test_main_keyboard_interrupt(monkeypatch):
  """Ensure keyboard interrupts exit with code 1."""
  monkeypatch.setattr(
    cli_module, "cli", lambda: (_ for _ in ()).throw(KeyboardInterrupt)
  )

  with pytest.raises(SystemExit) as exc:
    cli_module.main()

  assert exc.value.code == 1


def test_main_health_analyzer_error(monkeypatch):
  """Ensure HealthAnalyzerError exits with code 1."""
  monkeypatch.setattr(
    cli_module, "cli", lambda: (_ for _ in ()).throw(HealthAnalyzerError("boom"))
  )

  with pytest.raises(SystemExit) as exc:
    cli_module.main()

  assert exc.value.code == 1


def test_main_unexpected_error(monkeypatch):
  """Ensure unexpected errors exit with code 1."""
  monkeypatch.setattr(cli_module, "cli", lambda: (_ for _ in ()).throw(ValueError))
  logger = MagicMock()
  monkeypatch.setattr(cli_module, "logger", logger)

  with pytest.raises(SystemExit) as exc:
    cli_module.main()

  assert exc.value.code == 1
  assert logger.exception.called


def test_save_analysis_results_text_skips_optional_sections(tmp_path):
  """Ensure text export writes without optional fields."""
  report = SimpleNamespace(
    resting_hr_analysis=None,
    hrv_analysis=None,
    cardio_fitness=None,
    data_quality_score=0.9,
    record_count=5,
    quality_metrics=None,
  )
  highlights = SimpleNamespace(insights=[], recommendations=[])

  cli_module._save_analysis_results_text(tmp_path, report, report, highlights)

  assert (tmp_path / "analysis_results.txt").exists()


def test_save_parsed_data_empty_records(monkeypatch, tmp_path):
  """Ensure empty record list returns early."""
  console = MagicMock()
  console.print = MagicMock()
  monkeypatch.setattr(cli_module, "console", console)

  cli_module._save_parsed_data([], {}, tmp_path)

  assert console.print.called
