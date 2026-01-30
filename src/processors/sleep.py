"""Sleep data specialized analysis module.

Provides in-depth analysis of sleep-related data, including sleep stages, sleep quality, sleep patterns, etc.
"""

from dataclasses import dataclass
from datetime import (
  date,
  datetime,
  timedelta,
)
from typing import Any, Literal, cast

import pandas as pd

from ..analyzers.anomaly import AnomalyDetector
from ..analyzers.statistical import StatisticalAnalyzer
from ..core.data_models import CategoryRecord, HealthRecord, QuantityRecord
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SleepStage:
  """Sleep stage data"""

  stage: Literal["InBed", "Asleep", "Awake", "Core", "Deep", "REM"]
  start_date: datetime
  end_date: datetime
  duration_minutes: float


@dataclass
class SleepSession:
  """Single sleep session"""

  session_id: str
  start_date: datetime
  end_date: datetime
  total_duration: float  # Total duration (minutes)
  sleep_duration: float  # Actual sleep duration (minutes)
  awake_duration: float  # Awake duration (minutes)
  efficiency: float  # Sleep efficiency (0-1)

  # Sleep metrics
  sleep_latency: float = 0  # Sleep latency (minutes)
  wake_after_onset: float = 0  # Wake after sleep onset (minutes)
  awakenings_count: int = 0  # Number of awakenings

  # Sleep stage distribution
  core_sleep: float = 0  # Core sleep duration (minutes)
  deep_sleep: float = 0  # Deep sleep duration (minutes)
  rem_sleep: float = 0  # REM sleep duration (minutes)
  light_sleep: float = 0  # Light sleep duration (minutes)


@dataclass
class SleepQualityMetrics:
  """Sleep quality metrics"""

  average_duration: float  # Average sleep duration (hours)
  average_efficiency: float  # Average sleep efficiency (0-1)
  average_latency: float  # Average sleep latency (minutes)
  consistency_score: float  # Consistency score (0-1)
  overall_quality_score: float  # Overall quality score (0-100)

  # Sleep stage percentages
  core_sleep_percentage: float = 0
  deep_sleep_percentage: float = 0
  rem_sleep_percentage: float = 0
  light_sleep_percentage: float = 0


@dataclass
class SleepPatternAnalysis:
  """Sleep pattern analysis"""

  bedtime_consistency: float  # Bedtime consistency (0-1)
  waketime_consistency: float  # Waketime consistency (0-1)
  weekday_vs_weekend: dict[str, float]  # Weekday vs weekend comparison
  seasonal_patterns: dict[str, Any]  # Seasonal patterns

  # Trend analysis
  duration_trend: Literal["increasing", "decreasing", "stable"]
  efficiency_trend: Literal["improving", "declining", "stable"]


@dataclass
class SleepHeartRateCorrelation:
  """Sleep-heart rate correlation analysis"""

  avg_sleep_hr: float  # Average heart rate during sleep
  hr_variability_during_sleep: float  # Heart rate variability during sleep
  hr_drop_efficiency: float  # Heart rate drop efficiency when falling asleep
  recovery_quality: float  # Recovery quality score (based on HRV changes)


@dataclass
class SleepAnalysisReport:
  """Comprehensive sleep analysis report"""

  analysis_date: datetime
  data_range: tuple[datetime, datetime]

  # Core analysis
  quality_metrics: SleepQualityMetrics | None = None
  pattern_analysis: SleepPatternAnalysis | None = None
  hr_correlation: SleepHeartRateCorrelation | None = None

  # Detailed data
  sleep_sessions: list[SleepSession] | None = None
  daily_summary: pd.DataFrame | None = None
  weekly_summary: pd.DataFrame | None = None

  # Anomaly detection
  anomalies: list[Any] | None = None
  anomaly_report: dict[str, Any] | None = None

  # Trend analysis
  trends: dict[str, Any] | None = None

  # Highlights and recommendations
  highlights: list[str] | None = None
  recommendations: list[str] | None = None

  # Data quality
  data_quality_score: float = 0.0
  record_count: int = 0


class SleepAnalyzer:
  """Specialized sleep data analyzer

  Provides in-depth analysis of sleep-related data, including:
  - Sleep stage parsing and analysis
  - Sleep quality assessment
  - Sleep pattern recognition
  - Sleep-heart rate correlation analysis
  - Anomaly detection and health insights
  """

  def __init__(self):
    """Initialize sleep analyzer"""
    # Initialize analysis components.
    self.stat_analyzer = StatisticalAnalyzer()
    self.anomaly_detector = AnomalyDetector()

    logger.info("SleepAnalyzer initialized")

  def analyze_comprehensive(
    self,
    sleep_records: list[HealthRecord],
    heart_rate_records: list[HealthRecord] | None = None,
  ) -> SleepAnalysisReport:
    """Perform comprehensive sleep data analysis

    Args:
        sleep_records: Sleep records
        heart_rate_records: Heart rate records (for correlation analysis)

    Returns:
        Comprehensive analysis report
    """
    logger.info("Starting comprehensive sleep analysis")

    if not sleep_records:
      logger.warning("No sleep records provided for analysis")
      return SleepAnalysisReport(
        analysis_date=datetime.now(),
        data_range=(datetime.now(), datetime.now()),
      )

    data_range = self._calculate_data_range(sleep_records)
    analysis_date = datetime.now()

    # Parse sleep sessions.
    sleep_sessions = self._parse_sleep_sessions(sleep_records)

    if not sleep_sessions:
      logger.warning("No valid sleep sessions found")
      return SleepAnalysisReport(
        analysis_date=analysis_date,
        data_range=data_range,
      )

    # Analyze sleep quality.
    quality_metrics = self.analyze_sleep_quality(sleep_sessions)

    # Analyze sleep patterns.
    pattern_analysis = self.analyze_sleep_patterns(sleep_sessions)

    # Sleep-heart rate correlation analysis.
    hr_correlation = None
    if heart_rate_records:
      hr_correlation = self.analyze_sleep_hr_correlation(
        sleep_sessions, heart_rate_records
      )

    # Generate summary data.
    daily_summary = self._generate_daily_summary(sleep_sessions)
    weekly_summary = self._generate_weekly_summary(sleep_sessions)

    # Anomaly detection (based on duration and efficiency).
    anomalies = self._detect_sleep_anomalies(sleep_sessions)
    anomaly_report = {
      "total_sessions": len(sleep_sessions),
      "anomalies_detected": len(anomalies),
      "anomaly_rate": len(anomalies) / len(sleep_sessions) if sleep_sessions else 0,
    }

    # Trend analysis.
    trends = {}
    if not daily_summary.empty:
      duration_trend = self.stat_analyzer.analyze_trend(
        daily_summary, "date", "total_duration"
      )
      if duration_trend:
        trends["duration"] = duration_trend

      efficiency_trend = self.stat_analyzer.analyze_trend(
        daily_summary, "date", "efficiency"
      )
      if efficiency_trend:
        trends["efficiency"] = efficiency_trend

    # Generate highlights and recommendations.
    highlights = self._generate_highlights(
      quality_metrics, pattern_analysis, hr_correlation, trends, anomalies
    )
    recommendations = self._generate_recommendations(
      quality_metrics, pattern_analysis, hr_correlation, anomalies
    )

    # Data quality assessment.
    data_quality = self._assess_data_quality(sleep_records)

    report = SleepAnalysisReport(
      analysis_date=analysis_date,
      data_range=data_range,
      quality_metrics=quality_metrics,
      pattern_analysis=pattern_analysis,
      hr_correlation=hr_correlation,
      sleep_sessions=sleep_sessions,
      daily_summary=daily_summary,
      weekly_summary=weekly_summary,
      anomalies=anomalies,
      anomaly_report=anomaly_report,
      trends=trends,
      highlights=highlights,
      recommendations=recommendations,
      data_quality_score=data_quality,
      record_count=len(sleep_records),
    )

    logger.info("Comprehensive sleep analysis completed")
    return report

  def _parse_sleep_sessions(self, records: list[HealthRecord]) -> list[SleepSession]:
    """Parse sleep sessions from raw sleep records."""
    logger.info(f"Parsing {len(records)} sleep records into sessions")

    # Group records by date.
    records_by_date = {}
    for record in records:
      # Ensure start_date is a datetime object.
      if isinstance(record.start_date, datetime):
        date_key = record.start_date.date()
        if date_key not in records_by_date:
          records_by_date[date_key] = []
        records_by_date[date_key].append(record)

    sleep_sessions = []

    for date_key, day_records in records_by_date.items():
      try:
        session = self._parse_single_sleep_session(date_key, day_records)
        if session:
          sleep_sessions.append(session)
      except Exception as e:
        logger.warning(f"Failed to parse sleep session for {date_key}: {e}")
        continue

    logger.info(f"Parsed {len(sleep_sessions)} sleep sessions")
    return sleep_sessions

  def _parse_single_sleep_session(
    self, date: date, records: list[HealthRecord]
  ) -> SleepSession | None:
    """Parse a single sleep session."""
    # Filter sleep analysis records.
    sleep_records = [
      r for r in records if r.type == "HKCategoryTypeIdentifierSleepAnalysis"
    ]

    if not sleep_records:
      return None

    # Sort by time.
    sleep_records.sort(key=lambda r: r.start_date)

    # Identify distinct sleep sessions.
    # Apple Health groups continuous sleep records into sessions.
    # We pick the main session (typically the longest continuous period).

    # First, identify "InBed" records that define session boundaries.
    in_bed_records = [
      r
      for r in sleep_records
      if hasattr(r, "value")
      and isinstance(r, CategoryRecord)
      and str(r.value).endswith("InBed")
    ]

    if in_bed_records:
      # Use InBed records to define session boundaries.
      # Typically one InBed record corresponds to one session.
      main_bed_record = max(
        in_bed_records,
        key=lambda r: (r.end_date - r.start_date).total_seconds(),
      )
      session_start = main_bed_record.start_date
      # Use the latest end time as the session end.
      session_end = max(r.end_date for r in sleep_records)
      session_records = [
        r
        for r in sleep_records
        if r.start_date >= session_start and r.end_date <= session_end
      ]
    else:
      # If no InBed records exist, split sessions by time gaps.
      # Gaps over 2 hours are treated as separate sessions.
      sessions = []
      current_session = [sleep_records[0]]

      for i in range(1, len(sleep_records)):
        prev_end = sleep_records[i - 1].end_date
        curr_start = sleep_records[i].start_date
        gap = (curr_start - prev_end).total_seconds() / 3600  # Hours.

        if gap > 2:  # Gaps over 2 hours are separate sessions.
          sessions.append(current_session)
          current_session = [sleep_records[i]]
        else:
          current_session.append(sleep_records[i])

      sessions.append(current_session)

      # Pick the longest session (typically main nighttime sleep).
      if sessions:
        main_session = max(
          sessions,
          key=lambda s: sum((r.end_date - r.start_date).total_seconds() for r in s),
        )
        session_records = main_session
        session_start = min(r.start_date for r in session_records)
        session_end = max(r.end_date for r in session_records)
      else:
        session_records = sleep_records
        session_start = min(r.start_date for r in sleep_records)
        session_end = max(r.end_date for r in sleep_records)

    # Compute total duration (time in bed).
    total_duration = (session_end - session_start).total_seconds() / 60

    # Debug output.
    logger.debug(
      f"Session time range: {session_start} to {session_end}, total_duration={total_duration:.1f}min, records={len(session_records)}"
    )

    # Parse sleep stages.
    stages = []
    sleep_duration = 0
    awake_duration = 0
    core_sleep = 0
    deep_sleep = 0
    rem_sleep = 0
    light_sleep = 0

    # Debug: log the first few records.
    logger.debug(f"Debugging sleep records for {date} (total: {len(sleep_records)}):")
    for i, record in enumerate(sleep_records[:3]):  # Show only the first 3.
      logger.debug(
        f"  Record {i}: type={record.type}, value={getattr(record, 'value', 'N/A')}, "
        f"start={record.start_date}, end={record.end_date}"
      )

    for record in sleep_records:
      if hasattr(record, "value") and isinstance(record, CategoryRecord):
        stage_type = record.value
        duration = (record.end_date - record.start_date).total_seconds() / 60

        # Debug: log stage_type type and value.
        if len(sleep_records) <= 10:
          logger.debug(
            f"  Processing stage: raw_value={stage_type} (type: {type(stage_type)})"
          )

        # Apple Health stages are string values that need mapping.
        # From logs, formats look like:
        # HKCategoryValueSleepAnalysisAsleepCore -> Core
        # HKCategoryValueSleepAnalysisAsleepDeep -> Deep
        # HKCategoryValueSleepAnalysisAwake -> Awake
        # HKCategoryValueSleepAnalysisAsleepREM -> REM
        # HKCategoryValueSleepAnalysisAsleepUnspecified -> Asleep (or others)
        if isinstance(stage_type, str):
          if stage_type == "HKCategoryValueSleepAnalysisInBed":
            stage_type = "InBed"
          elif stage_type == "HKCategoryValueSleepAnalysisAwake":
            stage_type = "Awake"
          elif stage_type == "HKCategoryValueSleepAnalysisAsleepCore":
            stage_type = "Core"
          elif stage_type == "HKCategoryValueSleepAnalysisAsleepDeep":
            stage_type = "Deep"
          elif stage_type == "HKCategoryValueSleepAnalysisAsleepREM":
            stage_type = "REM"
          elif stage_type == "HKCategoryValueSleepAnalysisAsleepUnspecified":
            stage_type = "Asleep"  # Default to general sleep.
          # For unknown formats, try to extract the suffix.
          elif stage_type.startswith("HKCategoryValueSleepAnalysisAsleep"):
            # Extract suffix after "Asleep", e.g. "AsleepLight" -> "Light".
            suffix = stage_type.replace("HKCategoryValueSleepAnalysisAsleep", "")
            if suffix:
              stage_type = suffix
            else:
              stage_type = "Asleep"
          elif stage_type.startswith("HKCategoryValueSleepAnalysis"):
            stage_type = stage_type.replace("HKCategoryValueSleepAnalysis", "")
          # Preserve other strings.
        else:
          # For other types, convert to string.
          stage_type = str(stage_type)

        if len(sleep_records) <= 10:
          logger.debug(f"  Mapped stage: {stage_type}")

        # Type check: ensure stage_type is a valid stage.
        if stage_type in ["InBed", "Asleep", "Awake", "Core", "Deep", "REM"]:
          stages.append(
            SleepStage(
              stage=cast(
                Literal["InBed", "Asleep", "Awake", "Core", "Deep", "REM"],
                stage_type,
              ),
              start_date=record.start_date,
              end_date=record.end_date,
              duration_minutes=duration,
            )
          )

        # Sleep duration calculation:
        # - "Asleep" is generic and may overlap with specific stages.
        # - "Core", "Deep", "REM" are specific stages.
        # - Prefer specific stages; fall back to "Asleep" when needed.
        if stage_type in ["Core", "Deep", "REM"]:
          sleep_duration += duration
          if stage_type == "Core":
            core_sleep += duration
          elif stage_type == "Deep":
            deep_sleep += duration
          elif stage_type == "REM":
            rem_sleep += duration
        elif stage_type == "Asleep":
          # Use "Asleep" for light sleep only when no specific stages exist.
          # Simplified: count as light sleep if not overlapping.
          light_sleep += duration
          sleep_duration += duration
        elif stage_type == "Awake":
          awake_duration += duration
        elif stage_type == "InBed":
          # InBed is total time in bed, not counted in sleep duration.
          pass

    # Debug output.
    if len(sleep_records) <= 10:
      logger.debug(
        f"Session summary for {date}: total_duration={total_duration:.1f}min, "
        f"sleep_duration={sleep_duration:.1f}min, stages_count={len(stages)}"
      )

    # Compute sleep efficiency.
    efficiency = sleep_duration / total_duration if total_duration > 0 else 0

    # Compute sleep latency (time from bed to first sleep).
    sleep_latency = 0
    if stages:
      first_asleep = next(
        (s for s in stages if s.stage in ["Asleep", "Core", "Deep", "REM"]),
        None,
      )
      if first_asleep:
        sleep_latency = (first_asleep.start_date - session_start).total_seconds() / 60

    # Compute awakenings and awake duration.
    awake_stages = [s for s in stages if s.stage == "Awake"]
    awakenings_count = len(awake_stages)
    wake_after_onset = sum(s.duration_minutes for s in awake_stages)

    session_id = f"sleep_{date}_{session_start.strftime('%H%M')}"

    return SleepSession(
      session_id=session_id,
      start_date=session_start,
      end_date=session_end,
      total_duration=round(total_duration, 1),
      sleep_duration=round(sleep_duration, 1),
      awake_duration=round(awake_duration, 1),
      efficiency=round(efficiency, 3),
      core_sleep=round(core_sleep, 1),
      deep_sleep=round(deep_sleep, 1),
      rem_sleep=round(rem_sleep, 1),
      light_sleep=round(light_sleep, 1),
      sleep_latency=round(sleep_latency, 1),
      wake_after_onset=round(wake_after_onset, 1),
      awakenings_count=awakenings_count,
    )

  def analyze_sleep_quality(
    self, sleep_sessions: list[SleepSession]
  ) -> SleepQualityMetrics:
    """Analyze sleep quality

    Args:
        sleep_sessions: List of sleep sessions

    Returns:
        Sleep quality metrics
    """
    if not sleep_sessions:
      return SleepQualityMetrics(
        average_duration=0,
        average_efficiency=0,
        average_latency=0,
        consistency_score=0,
        overall_quality_score=0,
      )

    logger.info(f"Analyzing sleep quality from {len(sleep_sessions)} sessions")

    # Basic metrics.
    durations = [s.total_duration for s in sleep_sessions]
    efficiencies = [s.efficiency for s in sleep_sessions]
    latencies = [s.sleep_latency for s in sleep_sessions]

    average_duration = sum(durations) / len(durations) / 60  # Convert to hours.
    average_efficiency = sum(efficiencies) / len(efficiencies)
    average_latency = sum(latencies) / len(latencies)

    # Consistency scores (coefficient of variation: std/mean).
    duration_series = pd.Series(durations)
    efficiency_series = pd.Series(efficiencies)
    latency_series = pd.Series(latencies)

    duration_cv = (
      duration_series.std() / duration_series.mean()
      if duration_series.mean() > 0
      else float("inf")
    )
    efficiency_cv = (
      efficiency_series.std() / efficiency_series.mean()
      if efficiency_series.mean() > 0
      else float("inf")
    )
    latency_cv = (
      latency_series.std() / latency_series.mean()
      if latency_series.mean() > 0
      else float("inf")
    )

    # CV in 0-1 is acceptable; >1 indicates high variability.
    duration_consistency = max(0, min(1, 1 - duration_cv))
    efficiency_consistency = max(0, min(1, 1 - efficiency_cv))
    latency_consistency = max(0, min(1, 1 - latency_cv))

    consistency_score = (
      duration_consistency + efficiency_consistency + latency_consistency
    ) / 3

    # Stage percentages (sessions with stage data only).
    sessions_with_stages = [
      s
      for s in sleep_sessions
      if (s.core_sleep + s.deep_sleep + s.rem_sleep + s.light_sleep) > 0
    ]

    if sessions_with_stages:
      avg_core = sum(s.core_sleep for s in sessions_with_stages) / len(
        sessions_with_stages
      )
      avg_deep = sum(s.deep_sleep for s in sessions_with_stages) / len(
        sessions_with_stages
      )
      avg_rem = sum(s.rem_sleep for s in sessions_with_stages) / len(
        sessions_with_stages
      )
      avg_light = sum(s.light_sleep for s in sessions_with_stages) / len(
        sessions_with_stages
      )

      total_sleep = avg_core + avg_deep + avg_rem + avg_light
      if total_sleep > 0:
        core_sleep_percentage = avg_core / total_sleep
        deep_sleep_percentage = avg_deep / total_sleep
        rem_sleep_percentage = avg_rem / total_sleep
        light_sleep_percentage = avg_light / total_sleep
      else:
        core_sleep_percentage = deep_sleep_percentage = rem_sleep_percentage = (
          light_sleep_percentage
        ) = 0
    else:
      core_sleep_percentage = deep_sleep_percentage = rem_sleep_percentage = (
        light_sleep_percentage
      ) = 0

    # Overall quality score (0-100).
    # Based on duration (25%), efficiency (25%), consistency (25%), stages (25%).
    duration_score = min(100, average_duration / 8 * 100)  # 8 hours is full score.
    efficiency_score = average_efficiency * 100
    consistency_score_100 = consistency_score * 100

    # Stage score based on deep sleep and REM percentages.
    stage_score = (
      deep_sleep_percentage * 40
      + rem_sleep_percentage * 30
      + core_sleep_percentage * 20
      + light_sleep_percentage * 10
    ) * 100

    overall_quality_score = (
      duration_score * 0.25
      + efficiency_score * 0.25
      + consistency_score_100 * 0.25
      + stage_score * 0.25
    )

    return SleepQualityMetrics(
      average_duration=round(average_duration, 1),
      average_efficiency=round(average_efficiency, 3),
      average_latency=round(average_latency, 1),
      consistency_score=round(consistency_score, 3),
      core_sleep_percentage=round(core_sleep_percentage, 3),
      deep_sleep_percentage=round(deep_sleep_percentage, 3),
      rem_sleep_percentage=round(rem_sleep_percentage, 3),
      light_sleep_percentage=round(light_sleep_percentage, 3),
      overall_quality_score=round(overall_quality_score, 1),
    )

  def analyze_sleep_patterns(
    self, sleep_sessions: list[SleepSession]
  ) -> SleepPatternAnalysis:
    """Analyze sleep patterns.

    Args:
        sleep_sessions: List of sleep sessions.

    Returns:
        Sleep pattern analysis results.
    """
    if not sleep_sessions:
      return SleepPatternAnalysis(
        bedtime_consistency=0,
        waketime_consistency=0,
        weekday_vs_weekend={},
        seasonal_patterns={},
        duration_trend="stable",
        efficiency_trend="stable",
      )

    logger.info(f"Analyzing sleep patterns from {len(sleep_sessions)} sessions")

    # Extract bedtimes and wake times.
    bedtimes = []
    waketimes = []
    weekdays_data = []
    weekends_data = []

    for session in sleep_sessions:
      bedtime = session.start_date.hour + session.start_date.minute / 60
      waketime = session.end_date.hour + session.end_date.minute / 60

      bedtimes.append(bedtime)
      waketimes.append(waketime)

      # Separate weekdays and weekends.
      if session.start_date.weekday() < 5:  # Monday to Friday.
        weekdays_data.append(
          {
            "bedtime": bedtime,
            "waketime": waketime,
            "duration": session.total_duration,
            "efficiency": session.efficiency,
          }
        )
      else:  # Saturday and Sunday.
        weekends_data.append(
          {
            "bedtime": bedtime,
            "waketime": waketime,
            "duration": session.total_duration,
            "efficiency": session.efficiency,
          }
        )

    # Compute consistency.
    bedtime_consistency = (
      1 - (pd.Series(bedtimes).std() / 6) if bedtimes else 0
    )  # 6-hour range.
    waketime_consistency = 1 - (pd.Series(waketimes).std() / 6) if waketimes else 0

    bedtime_consistency = max(0, min(1, bedtime_consistency))
    waketime_consistency = max(0, min(1, waketime_consistency))

    # Weekend vs weekday comparison.
    weekday_vs_weekend = {}

    if weekdays_data and weekends_data:
      weekday_avg = pd.DataFrame(weekdays_data).mean()
      weekend_avg = pd.DataFrame(weekends_data).mean()

      weekday_vs_weekend = {
        "bedtime_difference": weekend_avg["bedtime"] - weekday_avg["bedtime"],
        "waketime_difference": weekend_avg["waketime"] - weekday_avg["waketime"],
        "duration_difference": (weekend_avg["duration"] - weekday_avg["duration"])
        / 60,  # Hours.
        "social_jetlag": abs(
          weekend_avg["bedtime"] - weekday_avg["bedtime"]
        ),  # Social jetlag.
      }

    # Seasonal patterns (simplified).
    seasonal_patterns = self._analyze_seasonal_patterns(sleep_sessions)

    # Trend analysis (based on recent data).
    recent_sessions = sorted(sleep_sessions, key=lambda s: s.start_date, reverse=True)[
      :30
    ]  # Most recent 30 days.

    if len(recent_sessions) >= 7:
      recent_durations = [s.total_duration for s in recent_sessions[:7]]
      older_durations = [s.total_duration for s in recent_sessions[7:14]]

      if older_durations:
        duration_change = sum(recent_durations) / len(recent_durations) - sum(
          older_durations
        ) / len(older_durations)
        duration_trend = (
          "increasing"
          if duration_change > 30
          else "decreasing"
          if duration_change < -30
          else "stable"
        )
      else:
        duration_trend = "stable"
    else:
      duration_trend = "stable"

    # Efficiency trend.
    if len(recent_sessions) >= 7:
      recent_efficiency = [s.efficiency for s in recent_sessions[:7]]
      older_efficiency = [s.efficiency for s in recent_sessions[7:14]]

      if older_efficiency:
        efficiency_change = sum(recent_efficiency) / len(recent_efficiency) - sum(
          older_efficiency
        ) / len(older_efficiency)
        efficiency_trend = (
          "improving"
          if efficiency_change > 0.05
          else "declining"
          if efficiency_change < -0.05
          else "stable"
        )
      else:
        efficiency_trend = "stable"
    else:
      efficiency_trend = "stable"

    return SleepPatternAnalysis(
      bedtime_consistency=round(bedtime_consistency, 3),
      waketime_consistency=round(waketime_consistency, 3),
      weekday_vs_weekend=weekday_vs_weekend,
      seasonal_patterns=seasonal_patterns,
      duration_trend=duration_trend,
      efficiency_trend=efficiency_trend,
    )

  def analyze_sleep_hr_correlation(
    self,
    sleep_sessions: list[SleepSession],
    heart_rate_records: list[HealthRecord],
  ) -> SleepHeartRateCorrelation | None:
    """Analyze sleep-heart rate correlation.

    Args:
        sleep_sessions: List of sleep sessions.
        heart_rate_records: List of heart rate records.

    Returns:
        Sleep-heart rate correlation results.
    """
    if not sleep_sessions or not heart_rate_records:
      return None

    logger.info("Analyzing sleep-heart rate correlation")

    # Convert heart rate records to a DataFrame.
    hr_data = []
    for r in heart_rate_records:
      # Check for QuantityRecord or CategoryRecord with value attributes.
      if isinstance(r, (QuantityRecord, CategoryRecord)) and hasattr(r, "start_date"):
        hr_data.append(
          {
            "timestamp": r.start_date,
            "value": r.value,
          }
        )

    hr_df = pd.DataFrame(hr_data)

    hr_df = hr_df.dropna()
    hr_df = hr_df.sort_values("timestamp")

    if hr_df.empty:
      return None

    # Compute heart rate metrics per sleep session.
    sleep_hr_metrics = []

    for session in sleep_sessions:
      # Get heart rate data during this session.
      session_hr = hr_df[
        (hr_df["timestamp"] >= session.start_date)
        & (hr_df["timestamp"] <= session.end_date)
      ]

      if not session_hr.empty:
        avg_hr = session_hr["value"].mean()
        hr_std = session_hr["value"].std()
        min_hr = session_hr["value"].min()

        # Calculate heart rate drop efficiency around sleep onset.
        # Compare 1 hour before vs 1 hour after sleep start.
        pre_sleep_hr = hr_df[
          (hr_df["timestamp"] >= session.start_date - timedelta(hours=1))
          & (hr_df["timestamp"] < session.start_date)
        ]

        post_sleep_hr = hr_df[
          (hr_df["timestamp"] >= session.start_date)
          & (hr_df["timestamp"] <= session.start_date + timedelta(hours=1))
        ]

        hr_drop_efficiency = 0
        if not pre_sleep_hr.empty and not post_sleep_hr.empty:
          pre_avg = pre_sleep_hr["value"].mean()
          post_avg = post_sleep_hr["value"].mean()
          if pre_avg > 0:
            hr_drop_efficiency = (pre_avg - post_avg) / pre_avg

        sleep_hr_metrics.append(
          {
            "avg_hr": avg_hr,
            "hr_variability": hr_std,
            "min_hr": min_hr,
            "hr_drop_efficiency": hr_drop_efficiency,
          }
        )

    if not sleep_hr_metrics:
      return None

    # Compute averaged metrics.
    avg_sleep_hr = sum(m["avg_hr"] for m in sleep_hr_metrics) / len(sleep_hr_metrics)
    hr_variability = sum(m["hr_variability"] for m in sleep_hr_metrics) / len(
      sleep_hr_metrics
    )
    hr_drop_efficiency = sum(m["hr_drop_efficiency"] for m in sleep_hr_metrics) / len(
      sleep_hr_metrics
    )

    # Recovery quality score (based on HR variability and drop efficiency).
    recovery_quality = (hr_variability * 0.4 + hr_drop_efficiency * 0.6) * 100
    recovery_quality = max(0, min(100, recovery_quality))

    return SleepHeartRateCorrelation(
      avg_sleep_hr=round(avg_sleep_hr, 1),
      hr_variability_during_sleep=round(hr_variability, 1),
      hr_drop_efficiency=round(hr_drop_efficiency, 3),
      recovery_quality=round(recovery_quality, 1),
    )

  def _generate_daily_summary(self, sleep_sessions: list[SleepSession]) -> pd.DataFrame:
    """Generate daily sleep summary."""
    if not sleep_sessions:
      return pd.DataFrame()

    # Group by date.
    daily_data = {}
    for session in sleep_sessions:
      date = session.start_date.date()
      if date not in daily_data:
        daily_data[date] = []
      daily_data[date].append(session)

    # Create per-day summary rows.
    summary_rows = []
    for date, sessions in daily_data.items():
      # If multiple sessions in a day, pick the longest.
      main_session = max(sessions, key=lambda s: s.total_duration)

      summary_rows.append(
        {
          "date": date,
          "total_duration": main_session.total_duration,
          "sleep_duration": main_session.sleep_duration,
          "efficiency": main_session.efficiency,
          "latency": main_session.sleep_latency,
          "awakenings": main_session.awakenings_count,
          "deep_sleep": main_session.deep_sleep,
          "rem_sleep": main_session.rem_sleep,
        }
      )

    return pd.DataFrame(summary_rows)

  def _generate_weekly_summary(
    self, sleep_sessions: list[SleepSession]
  ) -> pd.DataFrame:
    """Generate weekly sleep summary."""
    daily_df = self._generate_daily_summary(sleep_sessions)

    if daily_df.empty:
      return pd.DataFrame()

    # Aggregate by week.
    daily_df["week"] = pd.to_datetime(daily_df["date"]).dt.to_period("W")

    weekly_summary = (
      daily_df.groupby("week")
      .agg(
        {
          "total_duration": ["count", "mean", "std"],
          "sleep_duration": "mean",
          "efficiency": "mean",
          "latency": "mean",
          "awakenings": "mean",
          "deep_sleep": "mean",
          "rem_sleep": "mean",
        }
      )
      .round(2)
    )

    # Normalize column names.
    weekly_summary.columns = [
      "days_recorded",
      "avg_duration",
      "duration_std",
      "avg_sleep_duration",
      "avg_efficiency",
      "avg_latency",
      "avg_awakenings",
      "avg_deep_sleep",
      "avg_rem_sleep",
    ]

    weekly_summary = weekly_summary.reset_index()

    return weekly_summary

  def _detect_sleep_anomalies(self, sleep_sessions: list[SleepSession]) -> list[Any]:
    """Detect sleep anomalies."""
    if not sleep_sessions:
      return []

    # Convert to a DataFrame for anomaly detection.
    df = pd.DataFrame(
      [
        {
          "start_date": s.start_date,
          "total_duration": s.total_duration,
          "efficiency": s.efficiency,
          "latency": s.sleep_latency,
        }
        for s in sleep_sessions
      ]
    )

    # Detect duration and efficiency anomalies.
    anomalies = []

    # AnomalyDetector expects HealthRecord objects; use stats directly.
    # Duration anomaly detection.
    duration_values = df["total_duration"].dropna()
    if len(duration_values) >= 3:
      duration_mean = duration_values.mean()
      duration_std = duration_values.std()
      if duration_std > 0:
        for _, row in df.iterrows():
          if pd.notna(row["total_duration"]):
            z_score = abs(row["total_duration"] - duration_mean) / duration_std
            if z_score > 3.0:  # Use 3-sigma threshold.
              anomalies.append(
                {
                  "timestamp": row["start_date"],
                  "value": row["total_duration"],
                  "expected_value": duration_mean,
                  "deviation": z_score,
                  "severity": "high"
                  if z_score > 5.0
                  else "medium"
                  if z_score > 3.5
                  else "low",
                  "method": "zscore_duration",
                  "confidence": min(1.0, z_score / 5.0),
                }
              )

    # Efficiency anomaly detection.
    efficiency_values = df["efficiency"].dropna()
    if len(efficiency_values) >= 3:
      efficiency_mean = efficiency_values.mean()
      efficiency_std = efficiency_values.std()
      if efficiency_std > 0:
        for _, row in df.iterrows():
          if pd.notna(row["efficiency"]):
            z_score = abs(row["efficiency"] - efficiency_mean) / efficiency_std
            if z_score > 3.0:  # Use 3-sigma threshold.
              anomalies.append(
                {
                  "timestamp": row["start_date"],
                  "value": row["efficiency"] * 100,  # Convert to percentage.
                  "expected_value": efficiency_mean * 100,
                  "deviation": z_score,
                  "severity": "high"
                  if z_score > 5.0
                  else "medium"
                  if z_score > 3.5
                  else "low",
                  "method": "zscore_efficiency",
                  "confidence": min(1.0, z_score / 5.0),
                }
              )

    return anomalies

  def _analyze_seasonal_patterns(
    self, sleep_sessions: list[SleepSession]
  ) -> dict[str, Any]:
    """Analyze seasonal patterns (simplified)."""
    if len(sleep_sessions) < 10:
      return {}

    # Group by month.
    monthly_data = {}
    for session in sleep_sessions:
      month = session.start_date.month
      if month not in monthly_data:
        monthly_data[month] = []
      monthly_data[month].append(session.total_duration)

    # Compute monthly averages.
    seasonal_patterns = {}
    for month, durations in monthly_data.items():
      if len(durations) >= 3:  # At least 3 days of data.
        seasonal_patterns[f"month_{month}"] = {
          "avg_duration": sum(durations) / len(durations),
          "count": len(durations),
        }

    return seasonal_patterns

  def _calculate_data_range(
    self, records: list[HealthRecord]
  ) -> tuple[datetime, datetime]:
    """Compute data time range."""
    if not records:
      now = datetime.now()
      return (now, now)

    start_dates = [r.start_date for r in records if hasattr(r, "start_date")]
    if not start_dates:
      now = datetime.now()
      return (now, now)

    start_date = min(start_dates)
    end_date = max(start_dates)

    return (start_date, end_date)

  def _generate_highlights(
    self,
    quality: SleepQualityMetrics | None,
    patterns: SleepPatternAnalysis | None,
    hr_corr: SleepHeartRateCorrelation | None,
    trends: dict[str, Any],
    anomalies: list[Any],
  ) -> list[str]:
    """Generate highlights."""
    highlights = []

    # Sleep quality highlights.
    if quality:
      duration_hours = quality.average_duration
      if duration_hours >= 7:
        highlights.append(f"😴 平均睡眠时长{duration_hours:.1f}小时，睡眠充足")
      elif duration_hours < 6:
        highlights.append(f"⚠️ 平均睡眠时长仅{duration_hours:.1f}小时，建议增加睡眠时间")

      efficiency_pct = quality.average_efficiency * 100
      if efficiency_pct >= 85:
        highlights.append(f"💤 睡眠效率{efficiency_pct:.0f}%，睡眠质量良好")
      else:
        highlights.append(f"⚠️ 睡眠效率仅{efficiency_pct:.0f}%，可能存在睡眠问题")

      if quality.consistency_score >= 0.7:
        highlights.append("📅 睡眠规律性良好，有助于身体恢复")
      else:
        highlights.append("⏰ 睡眠时间不规律，建议调整作息时间")

    # Sleep pattern highlights.
    if patterns:
      if patterns.bedtime_consistency >= 0.8:
        highlights.append("🌙 就寝时间很规律")
      if patterns.waketime_consistency >= 0.8:
        highlights.append("🌅 起床时间很规律")

      if patterns.weekday_vs_weekend.get("social_jetlag", 0) > 2:
        highlights.append("⚠️ 工作日和周末作息差异较大，可能影响生物钟")

    # Sleep-heart rate highlights.
    if hr_corr:
      if hr_corr.recovery_quality >= 80:
        highlights.append("💚 睡眠期间心率恢复良好，身体恢复状态佳")
      elif hr_corr.recovery_quality < 60:
        highlights.append("⚠️ 睡眠期间心率恢复不佳，建议关注压力管理")

    # Trend highlights.
    if trends:
      duration_trend_obj = trends.get("duration")
      if duration_trend_obj and hasattr(duration_trend_obj, "trend_direction"):
        duration_trend = duration_trend_obj.trend_direction
        if duration_trend == "increasing":
          highlights.append("📈 睡眠时长呈上升趋势")
        elif duration_trend == "decreasing":
          highlights.append("📉 睡眠时长呈下降趋势")

    # Anomaly highlights.
    if anomalies:
      anomaly_count = len(anomalies)
      if anomaly_count > 0:
        highlights.append(f"🔍 检测到{anomaly_count}个睡眠异常事件")

    return highlights

  def _generate_recommendations(
    self,
    quality: SleepQualityMetrics | None,
    patterns: SleepPatternAnalysis | None,
    hr_corr: SleepHeartRateCorrelation | None,
    anomalies: list[Any],
  ) -> list[str]:
    """Generate recommendations."""
    recommendations = []

    # Recommendations based on sleep quality.
    if quality:
      if quality.average_duration < 7:
        recommendations.append("建议每天保证7-9小时的睡眠时间")

      if quality.average_efficiency < 0.85:
        recommendations.append("改善睡眠环境：保持卧室凉爽、黑暗和安静")

      if quality.average_latency > 30:
        recommendations.append("建立睡前放松 routine，避免使用电子设备")

    # Recommendations based on sleep patterns.
    if patterns:
      # Compute overall consistency score.
      overall_consistency = (
        patterns.bedtime_consistency + patterns.waketime_consistency
      ) / 2
      if overall_consistency < 0.7:
        recommendations.append("保持规律的作息时间，包括周末")

      social_jetlag = patterns.weekday_vs_weekend.get("social_jetlag", 0)
      if social_jetlag > 2:
        recommendations.append("减少周末和工作日的作息差异，维持生物钟稳定")

    # Recommendations based on heart rate correlation.
    if hr_corr and hr_corr.recovery_quality < 70:
      recommendations.append("睡前避免剧烈运动和咖啡因，保持放松状态")

    # General recommendations.
    if not recommendations:
      recommendations.extend(
        [
          "保持规律的作息时间",
          "睡前2小时避免使用电子设备",
          "保持卧室适宜的温度和湿度",
        ]
      )

    return recommendations

  def _assess_data_quality(self, records: list[HealthRecord]) -> float:
    """Assess data quality."""
    if not records:
      return 0.0

    # Check record completeness.
    total_records = len(records)
    sleep_analysis_records = sum(
      1 for r in records if r.type == "HKCategoryTypeIdentifierSleepAnalysis"
    )

    # Proportion of sleep analysis records.
    completeness = sleep_analysis_records / total_records if total_records > 0 else 0

    # Check temporal continuity (regular logging).
    if records:
      dates = sorted({r.start_date.date() for r in records})
      if len(dates) > 1:
        date_diffs = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        avg_gap = sum(date_diffs) / len(date_diffs)
        continuity = max(0, 1 - avg_gap / 7)  # 7 days is full score.
      else:
        continuity = 0.5
    else:
      continuity = 0

    # Combined score.
    quality_score = (completeness + continuity) / 2

    return round(float(quality_score), 3)
