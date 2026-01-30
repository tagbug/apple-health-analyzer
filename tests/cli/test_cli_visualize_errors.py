"""Error handling coverage for visualize command."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from click.testing import CliRunner

from src.cli_visualize import visualize


def test_visualize_parser_failure(monkeypatch, tmp_path):
  """Parser failures should exit with error."""
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  monkeypatch.setattr(
    "src.cli_visualize.get_config",
    lambda: SimpleNamespace(output_dir=tmp_path),
  )
  monkeypatch.setattr(
    "src.cli_visualize.StreamingXMLParser",
    lambda *_: SimpleNamespace(
      parse_records=lambda **kwargs: (_ for _ in ()).throw(Exception("boom"))
    ),
  )

  runner = CliRunner()
  result = runner.invoke(visualize, [str(xml_path), "--charts", "all"])

  assert result.exit_code == 1
  assert "Error: boom" in result.output


def test_visualize_chart_generation_error(monkeypatch, tmp_path):
  """Chart generation errors should be reported and continue."""
  xml_path = tmp_path / "data.xml"
  xml_path.write_text("<HealthData></HealthData>")

  monkeypatch.setattr(
    "src.cli_visualize.get_config",
    lambda: SimpleNamespace(output_dir=tmp_path),
  )

  record = SimpleNamespace(type="HKQuantityTypeIdentifierHeartRate")
  parser_instance = SimpleNamespace(parse_records=lambda **kwargs: [record])
  monkeypatch.setattr(
    "src.cli_visualize.StreamingXMLParser", lambda *_: parser_instance
  )

  class ErrorChartGenerator:
    def plot_heart_rate_timeseries(self, *args, **kwargs):
      raise Exception("chart fail")

  monkeypatch.setattr("src.cli_visualize.ChartGenerator", ErrorChartGenerator)
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.heart_rate_to_df",
    lambda *_: __import__("pandas").DataFrame({"timestamp": [1], "value": [1]}),
  )
  monkeypatch.setattr(
    "src.visualization.data_converter.DataConverter.sample_data_for_performance",
    lambda df, *_: df,
  )

  runner = CliRunner()
  result = runner.invoke(
    visualize,
    [str(xml_path), "--charts", "heart_rate_timeseries", "--static"],
  )

  assert result.exit_code == 0
  assert "Failed to generate" in result.output
