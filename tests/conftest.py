"""Test fixtures shared across the suite."""

from pathlib import Path


def pytest_configure() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  fixtures_dir = repo_root / "tests" / "fixtures"
  export_fixture = fixtures_dir / "export.xml"
  example_fixture = fixtures_dir / "example.xml"

  export_target = repo_root / "export_data" / "export.xml"
  example_target = repo_root / "example" / "example.xml"

  if not export_target.exists():
    export_target.parent.mkdir(parents=True, exist_ok=True)
    if export_fixture.exists():
      export_target.write_text(
        export_fixture.read_text(encoding="utf-8"), encoding="utf-8"
      )
  if not example_target.exists():
    example_target.parent.mkdir(parents=True, exist_ok=True)
    if example_fixture.exists():
      example_target.write_text(
        example_fixture.read_text(encoding="utf-8"), encoding="utf-8"
      )
