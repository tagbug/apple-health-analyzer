# PLAN

## Goals
1. Systematically analyze existing src/ and tests/ code; produce phased summaries.
2. Improve analyzers and visualization modules for richer analysis dimensions, higher accuracy, and better performance; enhance chart interactivity.
3. Add or extend test modules (unit, integration, coverage) with meaningful assertions; keep or improve coverage.
4. Ask for clarification when blocked; stop and report if repeated failures occur.
5. Split tasks into manageable sub-tasks and track via TODO.
6. Produce documentation after each phase and confirm readiness for the next.
7. Final report including optimizations, remaining gaps, and README updates.

## Phase 0: Baseline understanding
- Review analyzers (statistical.py, anomaly.py, highlights.py, extended_analyzer.py).
- Review processors (heart_rate.py, sleep.py, optimized_processor.py).
- Review visualization (charts.py, reports.py, data_converter.py).
- Review tests coverage and gaps.

Deliverable: docs/PHASE_0_SUMMARY.md

## Phase 1: Analyzer improvements
- Replace placeholder heuristics with data-driven metrics where possible.
- Add multi-dimensional metrics:
  - HR: variability by context, recovery profiles, zone stability, rolling anomalies.
  - Sleep: stage distribution robustness, latency/wake-after-onset robustness, social jetlag trends.
  - Cross-domain: sleep-HR and activity correlations with confidence indicators.
- Performance:
  - Reuse OptimizedDataFrame and StatisticalAggregator for heavy computations.
  - Reduce repeated DataFrame conversions.
- Improve anomaly detection:
  - Context-aware thresholds and per-metric calibration.
  - Clear severity definitions, avoid false positives.

Deliverable: docs/PHASE_1_SUMMARY.md

## Phase 2: Visualization improvements
- Add richer interactive features:
  - Compare series (day/week/month overlays).
  - Confidence bands and reference ranges.
  - Cross-filtering (if feasible) or aligned subplots.
- Improve chart semantics:
  - Clear axis labels, units, and annotations.
  - Visual cues for anomalies and key events.
- Ensure static/interactive parity.

Deliverable: docs/PHASE_2_SUMMARY.md

## Phase 3: Tests and coverage
- Add tests for new metrics and edge cases.
- Ensure no test for test's sake; assertions should validate real logic.
- Add integration tests for report/visualization flows.
- Confirm coverage does not regress.

Deliverable: docs/PHASE_3_SUMMARY.md

## Phase 4: Finalization
- Final report describing improvements and remaining gaps.
- Update README to reflect new analysis dimensions and visualization features.

Deliverable: docs/FINAL_REPORT.md
