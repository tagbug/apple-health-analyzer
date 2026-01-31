# Phase 2 - i18n Design and Integration

## Scope Completed
- Added a lightweight i18n module with translation store and formatter.
- Added locale selection via config/env and runtime resolution.
- Localized report content (HTML/Markdown) and visualization chart text.
- Localized report/chart logging messages.
- Localized CLI outputs in `src/cli.py` and `src/cli_visualize.py`.
- Localized analyzer outputs and logs (highlights, statistical, anomaly, extended analyzer).

## Implementation Summary
### i18n Core
- Added `src/i18n/store.py` with `TRANSLATIONS`, `DEFAULT_LOCALE`, and `SUPPORTED_LOCALES`.
- Added `src/i18n/translator.py` with a simple `.t(key, **kwargs)` formatter and fallback to English.
- Added `src/i18n/locale.py` with `resolve_locale()` to derive locale from explicit input or config.

### Configuration
- Added `locale` to `src/config.py` with validation and `LOCALE` env support.

### Reports
- Updated `src/visualization/reports.py` to use translations for titles, sections, labels, and footers.
- HTML `lang` attribute now switches between `en` and `zh-CN`.
- Log messages in report generation use i18n keys.

### Charts
- Updated `src/visualization/charts.py` to translate titles, axis labels, legends, hover templates, and annotations.
- Added translated labels for metrics used in comprehensive dashboards and risk factors.
- Log messages for chart generation/saving use i18n keys.

## Locale Flow
1. `LOCALE` env or config value is validated in `Config`.
2. `resolve_locale()` selects explicit locale when provided; otherwise falls back to config/default.
3. `Translator` fetches strings with English fallback if missing.

## Known Gaps
- Tests for i18n still pending for Phase 3.

## Files Updated
- `src/i18n/__init__.py`
- `src/i18n/store.py`
- `src/i18n/translator.py`
- `src/i18n/locale.py`
- `src/config.py`
- `src/visualization/reports.py`
- `src/visualization/charts.py`
- `src/cli_visualize.py`
- `src/cli.py`
- `src/analyzers/highlights.py`
- `src/analyzers/statistical.py`
- `src/analyzers/anomaly.py`
- `src/analyzers/extended_analyzer.py`
- `docs/TODO.md`

## Next Steps
- Continue localization for remaining processors and utils (exporter, benchmark, validator, optimized_processor, cleaner, logger).
- Add i18n tests (unit + integration) in Phase 3.
