"""Enhanced chart generation module - provides advanced health data visualization charts"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from ..i18n import Translator, resolve_locale
from ..processors.heart_rate import HeartRateAnalysisReport
from ..processors.sleep import SleepAnalysisReport
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Health-themed color palette
HEALTH_COLORS = {
  "primary": "#4CAF50",  # Health green
  "secondary": "#2196F3",  # Info blue
  "warning": "#FF9800",  # Orange
  "danger": "#F44336",  # Red
  "neutral": "#9E9E9E",  # Gray
  "success": "#8BC34A",  # Light green
  "info": "#03A9F4",  # Light blue
  "light": "#E8F5E9",  # Light background
  "dark": "#1B5E20",  # Dark green
}

# Plotly theme configuration
PLOTLY_TEMPLATE = {
  "layout": {
    "font": {"family": "Arial, sans-serif", "size": 12},
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "colorway": [
      HEALTH_COLORS["primary"],
      HEALTH_COLORS["secondary"],
      HEALTH_COLORS["warning"],
      HEALTH_COLORS["success"],
      HEALTH_COLORS["info"],
    ],
    "hovermode": "x unified",
  }
}


class ChartGenerator:
  """Chart generation core class

  Provides various health data visualization chart generation capabilities, supporting both interactive and static charts.
  """

  def __init__(
    self,
    theme: str = "health",
    width: int = 1200,
    height: int = 600,
    dpi: int = 300,
    locale: str | None = None,
  ):
    """Initialize chart generator.

    Args:
        theme: Theme name (health/light/dark).
        width: Chart width.
        height: Chart height.
        dpi: Chart DPI (for static image export).
    """
    self.theme = theme
    self.width = width
    self.height = height
    self.dpi = dpi
    self.translator = Translator(resolve_locale(locale))

    # Configure matplotlib styling.
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    logger.info(
      self.translator.t(
        "log.chart_generator_initialized",
        theme=theme,
        width=width,
        height=height,
      )
    )

  def plot_heart_rate_timeseries(
    self,
    data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
    interactive: bool = True,
    sleep_sessions: list[Any] | None = None,
  ) -> go.Figure | None:
    """Plot heart rate timeseries.

    Args:
        data: DataFrame with timestamp/value columns.
        title: Chart title.
        output_path: Output path (optional).
        interactive: Whether to generate interactive charts.

    Returns:
        Plotly Figure or None.
    """
    if data.empty:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.heart_rate_timeseries"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating",
        chart=self.translator.t("chart.title.heart_rate_timeseries"),
        count=len(data),
      )
    )
    chart_title = title or self.translator.t("chart.title.heart_rate_timeseries")

    # Downsample if the dataset is large.
    if len(data) > 10000:
      logger.info(
        self.translator.t(
          "log.chart_downsampling",
          source=len(data),
          target=10000,
        )
      )
      data = self._downsample_data(data, 10000)

    try:
      if interactive:
        # Generate interactive chart with Plotly.
        fig = go.Figure()

        data_sorted = data.sort_values("timestamp")
        if len(data_sorted) > 1:
          resample_rule = "h" if len(data_sorted) > 2000 else "30min"
          resampled = (
            data_sorted.set_index("timestamp")["value"]
            .resample(resample_rule)
            .mean()
            .dropna()
            .reset_index()
          )
        else:
          resampled = data_sorted

        # Background shading for sleep sessions.
        merged_sleep_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if sleep_sessions:
          sorted_sessions = sorted(sleep_sessions, key=lambda s: s.start_date)
          for session in sorted_sessions:
            start = pd.Timestamp(session.start_date)
            end = pd.Timestamp(session.end_date)
            if not merged_sleep_intervals:
              merged_sleep_intervals.append((start, end))
              continue
            last_start, last_end = merged_sleep_intervals[-1]
            if start <= last_end:
              merged_sleep_intervals[-1] = (last_start, max(last_end, end))
            else:
              merged_sleep_intervals.append((start, end))

          if len(merged_sleep_intervals) <= 200:
            for start, end in merged_sleep_intervals:
              fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="#0A84FF",
                opacity=0.06,
                line_width=0,
                layer="below",
              )

        # Smoothed trend line.
        fig.add_trace(
          go.Scatter(
            x=resampled["timestamp"],
            y=resampled["value"],
            mode="lines",
            name=self.translator.t("chart.label.heart_rate_trend"),
            line={"color": HEALTH_COLORS["neutral"], "width": 2},
            hovertemplate=f"<b>{self.translator.t('chart.label.time')}</b>: %{{x}}<br>"
            + f"<b>{self.translator.t('chart.label.heart_rate')}</b>: %{{y:.0f}} bpm<br>"
            + "<extra></extra>",
          )
        )

        if not resampled.empty:
          if sleep_sessions and merged_sleep_intervals:
            sleep_intervals = pd.IntervalIndex.from_tuples(
              merged_sleep_intervals,
              closed="both",
            )
            ts_index = pd.Index(pd.to_datetime(resampled["timestamp"]).to_numpy())
            idx, _ = sleep_intervals.get_indexer_non_unique(ts_index)
            resampled["is_sleep"] = idx >= 0
          else:
            resampled["hour"] = pd.to_datetime(resampled["timestamp"]).dt.hour
            resampled["is_sleep"] = (resampled["hour"] >= 22) | (resampled["hour"] < 6)

          def _hr_bucket(value: float, is_sleep: bool) -> str:
            if is_sleep:
              if value < 45:
                return "sleep_low"
              if value > 90:
                return "sleep_high"
              return "sleep_normal"
            if value < 50:
              return "wake_low"
            if value > 120:
              return "wake_high"
            return "wake_normal"

          resampled["bucket"] = resampled.apply(
            lambda row: _hr_bucket(row["value"], bool(row["is_sleep"])), axis=1
          )

          bucket_colors = {
            "sleep_low": "#64D2FF",
            "sleep_normal": "#0A84FF",
            "sleep_high": "#FF9F0A",
            "wake_low": "#5AC8FA",
            "wake_normal": "#34C759",
            "wake_high": "#FF453A",
          }
          bucket_labels = {
            "sleep_low": self.translator.t("chart.label.sleep_low"),
            "sleep_normal": self.translator.t("chart.label.sleep_normal"),
            "sleep_high": self.translator.t("chart.label.sleep_high"),
            "wake_low": self.translator.t("chart.label.wake_low"),
            "wake_normal": self.translator.t("chart.label.wake_normal"),
            "wake_high": self.translator.t("chart.label.wake_high"),
          }

          for bucket, subset in resampled.groupby("bucket"):
            fig.add_trace(
              go.Scatter(
                x=subset["timestamp"],
                y=subset["value"],
                mode="markers",
                name=bucket_labels.get(str(bucket), str(bucket)),
                marker={
                  "size": 7,
                  "color": bucket_colors.get(str(bucket), HEALTH_COLORS["neutral"]),
                  "opacity": 0.85,
                },
                hovertemplate=f"<b>{self.translator.t('chart.label.time')}</b>: %{{x}}<br>"
                + f"<b>{self.translator.t('chart.label.heart_rate')}</b>: %{{y:.0f}} bpm<br>"
                + "<extra></extra>",
              )
            )

        fig.update_layout(
          title=chart_title,
          xaxis_title=self.translator.t("chart.label.time"),
          yaxis_title=self.translator.t("chart.label.heart_rate_bpm"),
          width=self.width,
          height=self.height,
          template=PLOTLY_TEMPLATE,
          hovermode="x unified",
          legend={"orientation": "h", "y": -0.2},
        )

        if output_path:
          self._save_plotly_figure(fig, output_path)

        return fig
      else:
        # Generate static chart with Matplotlib.
        fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100))
        ax.plot(
          data["timestamp"],
          data["value"],
          color=HEALTH_COLORS["primary"],
          linewidth=1.5,
        )
        ax.axhline(
          data["value"].mean(),
          color=HEALTH_COLORS["secondary"],
          linestyle="--",
          label=f"{self.translator.t('chart.label.mean_value')}: "
          f"{data['value'].mean():.0f} bpm",
        )
        ax.set_xlabel(self.translator.t("chart.label.time"))
        ax.set_ylabel(self.translator.t("chart.label.heart_rate_bpm"))
        ax.set_title(chart_title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_path:
          plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
          logger.info(self.translator.t("log.chart_saved", path=output_path))

        plt.close()
        return None

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.heart_rate_timeseries"),
          error=str(e),
        )
      )
      return None

  def plot_resting_hr_trend(
    self,
    data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot resting heart rate trend.

    Args:
        data: Resting HR data (start_date/value columns).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if data.empty:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.resting_hr_trend"),
        )
      )
      return None

    date_col = "start_date" if "start_date" in data.columns else "timestamp"
    if date_col not in data.columns:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.resting_hr_trend"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating",
        chart=self.translator.t("chart.title.resting_hr_trend"),
        count=len(data),
      )
    )
    chart_title = title or self.translator.t("chart.title.resting_hr_trend")

    try:
      fig = go.Figure()

      # Actual data points.
      data_sorted = data.sort_values(date_col)
      fig.add_trace(
        go.Scatter(
          x=data_sorted[date_col],
          y=data_sorted["value"],
          mode="markers+lines",
          name=self.translator.t("report.section.resting_hr"),
          marker={
            "size": 6,
            "color": data_sorted["value"].apply(
              lambda value: HEALTH_COLORS["danger"]
              if value > 90
              else HEALTH_COLORS["info"]
              if value < 50
              else HEALTH_COLORS["success"]
            ),
            "opacity": 0.8,
          },
          line={"color": HEALTH_COLORS["primary"], "width": 2},
          hovertemplate=f"%{{x}}<br>{self.translator.t('chart.label.resting_hr_bpm')}: %{{y:.0f}} bpm<extra></extra>",
        )
      )

      # Add trend line (moving average).
      if len(data_sorted) > 7:
        ma_7 = data_sorted["value"].rolling(window=7, center=True).mean()
        fig.add_trace(
          go.Scatter(
            x=data_sorted[date_col],
            y=ma_7,
            mode="lines",
            name=self.translator.t("chart.label.moving_avg_7d"),
            line={
              "color": HEALTH_COLORS["secondary"],
              "width": 2,
              "dash": "dash",
            },
            hovertemplate=f"%{{x}}<br>{self.translator.t('chart.label.moving_avg_7d')}: %{{y:.0f}} bpm<extra></extra>",
          )
        )

      # Add healthy range band.
      fig.add_hrect(
        y0=60,
        y1=100,
        fillcolor=HEALTH_COLORS["success"],
        opacity=0.1,
        layer="below",
        line_width=0,
        annotation_text=self.translator.t("chart.label.healthy_range"),
        annotation_position="right",
      )

      # Add mean +/- std reference band.
      if "value" in data.columns and len(data) >= 3:
        mean_val = float(data["value"].mean())
        std_val = float(data["value"].std())
        if np.isfinite(mean_val) and np.isfinite(std_val) and std_val > 0:
          fig.add_hrect(
            y0=mean_val - std_val,
            y1=mean_val + std_val,
            fillcolor=HEALTH_COLORS["info"],
            opacity=0.08,
            layer="below",
            line_width=0,
            annotation_text=self.translator.t("chart.label.reference_band"),
            annotation_position="right",
          )

      fig.update_layout(
        title=chart_title,
        xaxis_title=self.translator.t("chart.label.date"),
        yaxis_title=self.translator.t("chart.label.resting_hr_bpm"),
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
      )

      fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector={
          "buttons": [
            {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
            {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
            {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
            {"step": "all", "label": "All"},
          ]
        },
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.resting_hr_trend"),
          error=str(e),
        )
      )
      return None

  def plot_hrv_analysis(
    self,
    data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot HRV analysis chart.

    Args:
        data: HRV data (columns: start_date, value).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if data.empty:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.hrv_analysis"),
        )
      )
      return None

    date_col = "start_date" if "start_date" in data.columns else "timestamp"
    if date_col not in data.columns:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.hrv_analysis"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating",
        chart=self.translator.t("chart.title.hrv_analysis"),
        count=len(data),
      )
    )
    chart_title = title or self.translator.t("chart.title.hrv_analysis")

    try:
      # Create subplots: HRV trend + distribution histogram.
      fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
          self.translator.t("chart.subtitle.hrv_trend"),
          self.translator.t("chart.subtitle.hrv_distribution"),
        ),
        vertical_spacing=0.15,
      )

      # HRV timeseries.
      data_sorted = data.sort_values(date_col)
      fig.add_trace(
        go.Scatter(
          x=data_sorted[date_col],
          y=data_sorted["value"],
          mode="lines+markers",
          name=self.translator.t("chart.label.sdnn_ms"),
          line={"color": HEALTH_COLORS["info"], "width": 2},
          marker={"size": 4},
          hovertemplate=f"%{{x}}<br>{self.translator.t('chart.label.sdnn_ms')}: %{{y:.0f}} ms<extra></extra>",
        ),
        row=1,
        col=1,
      )

      if len(data_sorted) >= 5:
        low_threshold = data_sorted["value"].quantile(0.2)
        low_mask = data_sorted["value"] <= low_threshold
        if low_mask.any():
          fig.add_trace(
            go.Scatter(
              x=data_sorted.loc[low_mask, date_col],
              y=data_sorted.loc[low_mask, "value"],
              mode="markers",
              name=self.translator.t("chart.label.hrv_low"),
              marker={
                "size": 7,
                "color": HEALTH_COLORS["danger"],
                "symbol": "x",
              },
              hovertemplate=(
                f"{self.translator.t('chart.label.sdnn_ms')}: %{{y:.0f}}<extra></extra>"
              ),
            ),
            row=1,
            col=1,
          )

      # Add moving average.
      if len(data_sorted) > 7:
        ma = data_sorted["value"].rolling(window=7, center=True).mean()
        fig.add_trace(
          go.Scatter(
            x=data_sorted[date_col],
            y=ma,
            mode="lines",
            name=self.translator.t("chart.label.moving_avg_7d"),
            line={
              "color": HEALTH_COLORS["secondary"],
              "width": 2,
              "dash": "dash",
            },
          ),
          row=1,
          col=1,
        )

      # HRV distribution histogram.
      fig.add_trace(
        go.Histogram(
          x=data["value"],
          name=self.translator.t("chart.label.distribution"),
          marker_color=HEALTH_COLORS["info"],
          opacity=0.7,
          nbinsx=30,
          hovertemplate=f"{self.translator.t('chart.label.sdnn_ms')}: %{{x:.0f}} ms<br>{self.translator.t('chart.label.frequency')}: %{{y}}<extra></extra>",
        ),
        row=2,
        col=1,
      )

      # Update layout.
      fig.update_xaxes(title_text=self.translator.t("chart.label.date"), row=1, col=1)
      fig.update_yaxes(
        title_text=self.translator.t("chart.label.sdnn_ms"), row=1, col=1
      )
      fig.update_xaxes(
        title_text=self.translator.t("chart.label.sdnn_ms"), row=2, col=1
      )
      fig.update_yaxes(
        title_text=self.translator.t("chart.label.frequency"), row=2, col=1
      )

      fig.update_layout(
        title=chart_title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        showlegend=True,
      )

      fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector={
          "buttons": [
            {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
            {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
            {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
            {"step": "all", "label": "All"},
          ]
        },
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.hrv_analysis"),
          error=str(e),
        )
      )
      return None

  def plot_heart_rate_heatmap(
    self,
    data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot heart rate heatmap (calendar view).

    Args:
        data: Daily heart rate data (columns: date, avg_hr).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if data.empty:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.heart_rate_heatmap"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating",
        chart=self.translator.t("chart.title.heart_rate_heatmap"),
        count=len(data),
      )
    )
    chart_title = title or self.translator.t("chart.title.heart_rate_heatmap")

    try:
      # Prepare heatmap data: week x weekday.
      data = data.copy()
      data["date"] = pd.to_datetime(data["date"])
      data["week"] = data["date"].dt.isocalendar().week
      data["weekday"] = data["date"].dt.dayofweek
      data["year"] = data["date"].dt.year

      # Create pivot table.
      pivot = data.pivot_table(
        values="avg_hr", index="week", columns="weekday", aggfunc="mean"
      )

      # Create heatmap.
      fig = go.Figure(
        data=go.Heatmap(
          z=pivot.values,
          x=[
            self.translator.t("chart.weekday.mon"),
            self.translator.t("chart.weekday.tue"),
            self.translator.t("chart.weekday.wed"),
            self.translator.t("chart.weekday.thu"),
            self.translator.t("chart.weekday.fri"),
            self.translator.t("chart.weekday.sat"),
            self.translator.t("chart.weekday.sun"),
          ],
          y=pivot.index,
          colorscale="RdYlGn_r",  # Red-yellow-green (reversed).
          colorbar={"title": f"{self.translator.t('chart.label.mean_value')}<br>(bpm)"},
          hovertemplate=f"<b>{self.translator.t('chart.label.week_number')} %{{y}} "
          f"%{{x}}</b><br>"
          + f"{self.translator.t('chart.label.mean_value')}: %{{z:.0f}} bpm<br>"
          + "<extra></extra>",
        )
      )

      fig.update_layout(
        title=chart_title,
        xaxis_title=self.translator.t("chart.label.weekday"),
        yaxis_title=self.translator.t("chart.label.week_number"),
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.heart_rate_heatmap"),
          error=str(e),
        )
      )
      return None

  def plot_heart_rate_distribution(
    self,
    data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot heart rate histogram and box plot.

    Args:
        data: Heart rate data (columns: value).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if data.empty or "value" not in data.columns:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.heart_rate_distribution"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating",
        chart=self.translator.t("chart.title.heart_rate_distribution"),
        count=len(data),
      )
    )
    chart_title = title or self.translator.t("chart.title.heart_rate_distribution")

    try:
      # Create subplots: histogram + box plot.
      fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=(
          self.translator.t("chart.subtitle.heart_rate_hist"),
          self.translator.t("chart.label.daily_weekly_monthly"),
        ),
        vertical_spacing=0.15,
      )

      # Prepare zone bins.
      max_hr = 200
      bins = [0, max_hr * 0.6, max_hr * 0.7, max_hr * 0.8, max_hr * 0.9, max_hr * 1.0]
      bin_labels = [
        self.translator.t("report.metric.hr_zone1"),
        self.translator.t("report.metric.hr_zone2"),
        self.translator.t("report.metric.hr_zone3"),
        self.translator.t("report.metric.hr_zone4"),
        self.translator.t("report.metric.hr_zone5"),
      ]
      hist, edges = np.histogram(data["value"].dropna(), bins=bins)
      # Zone-colored histogram.
      fig.add_trace(
        go.Bar(
          x=bin_labels,
          y=hist,
          marker_color=[
            "#34C759",
            "#0A84FF",
            "#5AC8FA",
            "#FF9F0A",
            "#FF453A",
          ],
          name=self.translator.t("chart.label.distribution"),
          hovertemplate=f"%{{x}}<br>{self.translator.t('chart.label.frequency')}: %{{y}}<extra></extra>",
        ),
        row=1,
        col=1,
      )

      # Daily summary histogram.
      data_copy = data.copy()
      if "start_date" in data_copy.columns:
        data_copy["date"] = pd.to_datetime(data_copy["start_date"]).dt.date
      elif "timestamp" in data_copy.columns:
        data_copy["date"] = pd.to_datetime(data_copy["timestamp"]).dt.date
      else:
        data_copy["date"] = pd.date_range(
          "1970-01-01", periods=len(data_copy), freq="D"
        )

      daily_mean = data_copy.groupby("date")["value"].mean().dropna()
      daily_mean = data_copy.groupby("date")["value"].mean().dropna()
      weekly_mean = (
        data_copy.set_index(pd.to_datetime(data_copy["date"]))
        .resample("W")["value"]
        .mean()
        .dropna()
      )
      monthly_mean = (
        data_copy.set_index(pd.to_datetime(data_copy["date"]))
        .resample("ME")["value"]
        .mean()
        .dropna()
      )

      fig.add_trace(
        go.Box(
          y=daily_mean,
          name=self.translator.t("chart.label.daily_mean"),
          marker_color="#0A84FF",
          boxmean=True,
          hovertemplate=f"{self.translator.t('chart.label.heart_rate_bpm')}: %{{y:.0f}} bpm<extra></extra>",
        ),
        row=2,
        col=1,
      )
      fig.add_trace(
        go.Box(
          y=weekly_mean,
          name=self.translator.t("chart.label.weekly_mean"),
          marker_color="#34C759",
          boxmean=True,
          hovertemplate=f"{self.translator.t('chart.label.heart_rate_bpm')}: %{{y:.0f}} bpm<extra></extra>",
        ),
        row=2,
        col=1,
      )
      fig.add_trace(
        go.Box(
          y=monthly_mean,
          name=self.translator.t("chart.label.monthly_mean"),
          marker_color="#5AC8FA",
          boxmean=True,
          hovertemplate=f"{self.translator.t('chart.label.heart_rate_bpm')}: %{{y:.0f}} bpm<extra></extra>",
        ),
        row=2,
        col=1,
      )

      # Update layout.
      fig.update_yaxes(
        title_text=self.translator.t("chart.label.frequency"), row=1, col=1
      )
      fig.update_yaxes(
        title_text=self.translator.t("chart.label.heart_rate_bpm"), row=2, col=1
      )

      fig.update_layout(
        title=chart_title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        showlegend=True,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.heart_rate_distribution"),
          error=str(e),
        )
      )
      return None

  def plot_heart_rate_zones(
    self,
    data: pd.DataFrame,
    max_hr: float = 220,
    age: int = 30,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot heart rate zone distribution.

    Args:
        data: Heart rate data (columns: value).
        max_hr: Maximum heart rate (default: 220).
        age: Age (used to estimate max heart rate).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if data.empty or "value" not in data.columns:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.heart_rate_zones"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating",
        chart=self.translator.t("chart.title.heart_rate_zones"),
        count=len(data),
      )
    )
    chart_title = title or self.translator.t("chart.title.heart_rate_zones")

    try:
      # Compute age-adjusted maximum heart rate.
      max_hr_adjusted = max_hr - age

      # Define heart rate zones.
      zones = {
        self.translator.t("chart.zone.rest"): (
          0.5 * max_hr_adjusted,
          0.6 * max_hr_adjusted,
        ),
        self.translator.t("chart.zone.fat_burn"): (
          0.6 * max_hr_adjusted,
          0.7 * max_hr_adjusted,
        ),
        self.translator.t("chart.zone.aerobic"): (
          0.7 * max_hr_adjusted,
          0.8 * max_hr_adjusted,
        ),
        self.translator.t("chart.zone.anaerobic"): (
          0.8 * max_hr_adjusted,
          0.9 * max_hr_adjusted,
        ),
        self.translator.t("chart.zone.peak"): (
          0.9 * max_hr_adjusted,
          max_hr_adjusted,
        ),
      }

      # Count records per zone.
      zone_counts = {}
      for zone_name, (lower, upper) in zones.items():
        count = len(data[(data["value"] >= lower) & (data["value"] < upper)])
        zone_counts[zone_name] = count

      # Create pie chart.
      fig = go.Figure(
        data=[
          go.Pie(
            labels=list(zone_counts.keys()),
            values=list(zone_counts.values()),
            marker={
              "colors": [
                HEALTH_COLORS["success"],
                HEALTH_COLORS["primary"],
                HEALTH_COLORS["info"],
                HEALTH_COLORS["warning"],
                HEALTH_COLORS["danger"],
              ]
            },
            hovertemplate="<b>%{label}</b><br>"
            + f"{self.translator.t('chart.label.records')}: %{{value}}<br>"
            + f"{self.translator.t('chart.label.percent')}: %{{percent}}<br>"
            + "<extra></extra>",
          )
        ]
      )

      fig.update_layout(
        title=chart_title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.heart_rate_zones"),
          error=str(e),
        )
      )
      return None

  def plot_heart_rate_advanced_metrics(
    self,
    report: HeartRateAnalysisReport,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot advanced heart rate metrics from the report.

    Args:
        report: Heart rate analysis report.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if not report.advanced_metrics:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.heart_rate_advanced"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating_generic",
        chart=self.translator.t("chart.title.heart_rate_advanced"),
      )
    )

    metrics = report.advanced_metrics
    diurnal = metrics.get("diurnal_profile", {})
    variability = metrics.get("variability", {})
    trends = metrics.get("recent_trends", {})
    zones = metrics.get("zones", {})

    try:
      fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
          self.translator.t("chart.label.diurnal_profile"),
          self.translator.t("report.metric.daily_cv"),
          self.translator.t("report.metric.hr_mean_7d"),
          self.translator.t("chart.label.zone_distribution"),
        ),
      )

      day_mean = diurnal.get("day_mean")
      night_mean = diurnal.get("night_mean")
      if day_mean is not None or night_mean is not None:
        fig.add_trace(
          go.Bar(
            x=[
              self.translator.t("chart.label.day"),
              self.translator.t("chart.label.night"),
            ],
            y=[day_mean or 0, night_mean or 0],
            marker_color=[HEALTH_COLORS["primary"], HEALTH_COLORS["info"]],
            name=self.translator.t("chart.label.diurnal_profile"),
            hovertemplate="<b>%{x}</b><br>%{y:.1f} bpm<extra></extra>",
          ),
          row=1,
          col=1,
        )

      daily_cv = variability.get("daily_cv_mean")
      if daily_cv is not None:
        fig.add_trace(
          go.Indicator(
            mode="number",
            value=float(daily_cv),
            title={"text": self.translator.t("report.metric.daily_cv")},
            number={"valueformat": ".3f"},
          ),
          row=1,
          col=2,
        )

      hr_mean_7d = trends.get("hr_mean_7d")
      hr_mean_30d = trends.get("hr_mean_30d")
      if hr_mean_7d is not None or hr_mean_30d is not None:
        fig.add_trace(
          go.Bar(
            x=["7d", "30d"],
            y=[hr_mean_7d or 0, hr_mean_30d or 0],
            marker_color=[HEALTH_COLORS["secondary"], HEALTH_COLORS["neutral"]],
            name=self.translator.t("report.metric.hr_mean_7d"),
            hovertemplate="<b>%{x}</b><br>%{y:.1f} bpm<extra></extra>",
          ),
          row=2,
          col=1,
        )

      if zones and zones.get("zone1"):
        zone_labels = [
          self.translator.t("report.metric.hr_zone1"),
          self.translator.t("report.metric.hr_zone2"),
          self.translator.t("report.metric.hr_zone3"),
          self.translator.t("report.metric.hr_zone4"),
          self.translator.t("report.metric.hr_zone5"),
        ]
        zone_values = [
          zones["zone1"]["percentage"],
          zones["zone2"]["percentage"],
          zones["zone3"]["percentage"],
          zones["zone4"]["percentage"],
          zones["zone5"]["percentage"],
        ]
        fig.add_trace(
          go.Bar(
            x=zone_labels,
            y=zone_values,
            marker_color=[
              HEALTH_COLORS["success"],
              HEALTH_COLORS["primary"],
              HEALTH_COLORS["info"],
              HEALTH_COLORS["warning"],
              HEALTH_COLORS["danger"],
            ],
            name=self.translator.t("report.metric.hr_zones"),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
          ),
          row=2,
          col=2,
        )

      fig.update_layout(
        title=self.translator.t("chart.title.heart_rate_advanced"),
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        showlegend=False,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.heart_rate_advanced"),
          error=str(e),
        )
      )
      return None

  def _downsample_data(self, data: pd.DataFrame, target_points: int) -> pd.DataFrame:
    """Downsample data using simple uniform sampling.

    Args:
        data: Source data.
        target_points: Target number of points.

    Returns:
        Downsampled data.
    """
    if len(data) <= target_points:
      return data

    step = len(data) // target_points
    return data.iloc[::step].copy()

  def _save_plotly_figure(self, fig: go.Figure, output_path: Path) -> None:
    """Save a Plotly figure.

    Args:
        fig: Plotly figure.
        output_path: Output path.
    """
    try:
      output_path = Path(output_path)
      output_path.parent.mkdir(parents=True, exist_ok=True)

      if output_path.suffix == ".html":
        fig.write_html(str(output_path))
      elif output_path.suffix in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        fig.write_image(str(output_path))
      else:
        # Default to HTML.
        fig.write_html(str(output_path.with_suffix(".html")))

      logger.info(self.translator.t("log.chart_saved", path=output_path))

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_save_error",
          path=output_path,
          error=str(e),
        )
      )

  def plot_sleep_timeline(
    self,
    data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot sleep timeline chart.

    Args:
        data: Sleep data (columns: start_date, end_date, value/stage).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if data.empty:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.sleep_timeline"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating",
        chart=self.translator.t("chart.title.sleep_timeline"),
        count=len(data),
      )
    )
    chart_title = title or self.translator.t("chart.title.sleep_timeline")

    try:
      # Sleep stage color mapping.
      stage_colors = {
        "Asleep": HEALTH_COLORS["primary"],
        "InBed": HEALTH_COLORS["light"],
        "Awake": HEALTH_COLORS["warning"],
        "Core": HEALTH_COLORS["info"],
        "Deep": HEALTH_COLORS["dark"],
        "REM": HEALTH_COLORS["secondary"],
      }

      fig = go.Figure()

      # Draw a segment for each sleep stage.
      for idx, row in data.iterrows():
        stage = row.get("value", "Asleep")
        if isinstance(stage, (int, float)):
          stage = "Asleep"

        fig.add_trace(
          go.Scatter(
            x=[row["start_date"], row["end_date"]],
            y=[idx, idx],
            mode="lines",
            line={
              "color": stage_colors.get(stage, HEALTH_COLORS["neutral"]),
              "width": 20,
            },
            name=stage,
            showlegend=False,
            hovertemplate=f"<b>{stage}</b><br>"
            + f"{self.translator.t('chart.label.start_time')}: {row['start_date']}<br>"
            + f"{self.translator.t('chart.label.end_time')}: {row['end_date']}<br>"
            + "<extra></extra>",
          )
        )

      fig.update_layout(
        title=chart_title,
        xaxis_title=self.translator.t("chart.label.time"),
        yaxis_title=self.translator.t("chart.label.sleep_session"),
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.sleep_timeline"),
          error=str(e),
        )
      )
      return None

  def plot_sleep_quality_trend(
    self,
    data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot sleep quality trend chart.

    Args:
        data: Sleep quality data (columns: date, duration, efficiency).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if data.empty:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.sleep_quality_trend"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating",
        chart=self.translator.t("chart.title.sleep_quality_trend"),
        count=len(data),
      )
    )
    chart_title = title or self.translator.t("chart.title.sleep_quality_trend")

    try:
      # Create a main chart + strip chart.
      fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(
          self.translator.t("chart.label.sleep_duration_hours"),
          self.translator.t("chart.label.sleep_efficiency_percent"),
        ),
      )

      # Sleep duration.
      duration_hours = data["total_duration"] / 60
      fig.add_trace(
        go.Scatter(
          x=data["date"],
          y=duration_hours,  # Convert to hours.
          mode="lines+markers",
          name=self.translator.t("chart.label.sleep_duration_hours"),
          line={"color": HEALTH_COLORS["primary"], "width": 2},
          marker={"size": 6},
          hovertemplate=f"%{{x}}<br>{self.translator.t('chart.label.sleep_duration_hours')}: %{{y:.1f}}<extra></extra>",
        ),
        row=1,
        col=1,
      )

      # Sleep efficiency.
      if "efficiency" in data.columns:
        efficiency_percent = data["efficiency"] * 100
        efficiency_colors = efficiency_percent.apply(
          lambda value: HEALTH_COLORS["danger"]
          if value < 80
          else HEALTH_COLORS["warning"]
          if value < 90
          else HEALTH_COLORS["success"]
        )
        fig.add_trace(
          go.Bar(
            x=data["date"],
            y=efficiency_percent,
            name=self.translator.t("chart.label.sleep_efficiency_percent"),
            marker_color=efficiency_colors,
            hovertemplate=f"%{{x}}<br>{self.translator.t('chart.label.sleep_efficiency_percent')}: %{{y:.0f}}%<extra></extra>",
          ),
          row=2,
          col=1,
        )

        anomaly_mask = (duration_hours < 6) | (data["efficiency"] < 0.8)
        if anomaly_mask.any():
          fig.add_trace(
            go.Scatter(
              x=data.loc[anomaly_mask, "date"],
              y=duration_hours[anomaly_mask],
              mode="markers",
              name=self.translator.t("chart.label.sleep_anomaly"),
              marker={
                "size": 10,
                "color": HEALTH_COLORS["danger"],
                "symbol": "x",
              },
              hovertemplate=(
                f"{self.translator.t('chart.label.sleep_duration_hours')}: %{{y:.1f}}<br>"
                + f"{self.translator.t('chart.label.sleep_efficiency_percent')}: %{{customdata:.0f}}%<extra></extra>"
              ),
              customdata=(data.loc[anomaly_mask, "efficiency"] * 100),
            ),
            row=1,
            col=1,
          )

      # Add recommended sleep duration line.
      fig.add_hline(
        y=7,
        line_dash="dash",
        line_color=HEALTH_COLORS["success"],
        annotation_text=self.translator.t("chart.label.recommended_sleep"),
        annotation_position="right",
      )

      # Add recommended sleep duration band (7-9 hours).
      fig.add_hrect(
        y0=7,
        y1=9,
        fillcolor=HEALTH_COLORS["success"],
        opacity=0.08,
        layer="below",
        line_width=0,
      )

      # Update layout.
      fig.update_xaxes(title_text=self.translator.t("chart.label.date"), row=2, col=1)
      fig.update_yaxes(
        title_text=self.translator.t("chart.label.sleep_duration_hours"), row=1, col=1
      )
      fig.update_yaxes(
        title_text=self.translator.t("chart.label.sleep_efficiency_percent"),
        row=2,
        col=1,
        range=[0, 100],
      )

      fig.update_layout(
        title=chart_title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
        showlegend=False,
      )

      fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector={
          "buttons": [
            {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
            {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
            {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
            {"step": "all", "label": "All"},
          ]
        },
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.sleep_quality_trend"),
          error=str(e),
        )
      )
      return None

  def plot_sleep_quality_details(
    self,
    data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot detailed sleep quality metrics in small panels.

    Args:
        data: Sleep quality data (columns: date, latency, wake_after_onset, awakenings).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if data.empty:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.sleep_quality_details"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating",
        chart=self.translator.t("chart.title.sleep_quality_details"),
        count=len(data),
      )
    )
    chart_title = title or self.translator.t("chart.title.sleep_quality_details")

    try:
      fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
          self.translator.t("report.metric.avg_sleep_latency"),
          self.translator.t("report.metric.avg_wake_after_onset"),
          self.translator.t("report.metric.avg_awakenings"),
        ),
        vertical_spacing=0.08,
      )

      if "latency" in data.columns:
        fig.add_trace(
          go.Scatter(
            x=data["date"],
            y=data["latency"],
            mode="lines+markers",
            name=self.translator.t("report.metric.avg_sleep_latency"),
            line={"color": HEALTH_COLORS["info"], "width": 2},
            marker={"size": 6},
            hovertemplate=f"%{{x}}<br>{self.translator.t('report.metric.avg_sleep_latency')}: %{{y:.1f}} {self.translator.t('report.metric.sleep_latency_unit')}<extra></extra>",
          ),
          row=1,
          col=1,
        )

      if "wake_after_onset" in data.columns:
        fig.add_trace(
          go.Scatter(
            x=data["date"],
            y=data["wake_after_onset"],
            mode="lines+markers",
            name=self.translator.t("report.metric.avg_wake_after_onset"),
            line={"color": HEALTH_COLORS["warning"], "width": 2},
            marker={"size": 6},
            hovertemplate=f"%{{x}}<br>{self.translator.t('report.metric.avg_wake_after_onset')}: %{{y:.1f}} {self.translator.t('report.metric.wake_after_onset_unit')}<extra></extra>",
          ),
          row=2,
          col=1,
        )

      if "awakenings" in data.columns:
        fig.add_trace(
          go.Scatter(
            x=data["date"],
            y=data["awakenings"],
            mode="lines+markers",
            name=self.translator.t("report.metric.avg_awakenings"),
            line={"color": HEALTH_COLORS["neutral"], "width": 2},
            marker={"size": 6},
            hovertemplate=f"%{{x}}<br>{self.translator.t('report.metric.avg_awakenings')}: %{{y:.1f}} {self.translator.t('report.metric.awakenings_unit')}<extra></extra>",
          ),
          row=3,
          col=1,
        )

      fig.update_xaxes(title_text=self.translator.t("chart.label.date"), row=3, col=1)
      fig.update_yaxes(
        title_text=self.translator.t("report.metric.sleep_latency_unit"), row=1, col=1
      )
      fig.update_yaxes(
        title_text=self.translator.t("report.metric.wake_after_onset_unit"),
        row=2,
        col=1,
      )
      fig.update_yaxes(
        title_text=self.translator.t("report.metric.awakenings_unit"), row=3, col=1
      )

      fig.update_layout(
        title=chart_title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        showlegend=False,
      )

      fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector={
          "buttons": [
            {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
            {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
            {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
            {"step": "all", "label": "All"},
          ]
        },
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.sleep_quality_details"),
          error=str(e),
        )
      )
      return None

  def plot_sleep_stages_distribution(
    self,
    data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot sleep stage distribution stacked by day.

    Args:
        data: Sleep stage data (columns: date, stage, duration).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if data.empty or not {"date", "stage", "duration"}.issubset(data.columns):
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.sleep_stages_distribution"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating_generic",
        chart=self.translator.t("chart.title.sleep_stages_distribution"),
      )
    )
    chart_title = title or self.translator.t("chart.title.sleep_stages_distribution")

    try:
      # Sleep stage colors.
      colors = {
        "Asleep": HEALTH_COLORS["primary"],
        "Core": HEALTH_COLORS["info"],
        "Deep": HEALTH_COLORS["dark"],
        "REM": HEALTH_COLORS["secondary"],
        "Light": HEALTH_COLORS["light"],
        "Awake": HEALTH_COLORS["warning"],
      }

      data_copy = data.copy()
      data_copy = data_copy.dropna(subset=["date", "stage", "duration"])
      if data_copy.empty:
        logger.warning(
          self.translator.t(
            "log.chart_no_data",
            chart=self.translator.t("chart.title.sleep_stages_distribution"),
          )
        )
        return None
      data_copy["date"] = pd.to_datetime(data_copy["date"]).dt.date
      pivot = data_copy.pivot_table(
        index="date", columns="stage", values="duration", aggfunc="sum"
      ).fillna(0)

      fig = go.Figure()
      for stage in pivot.columns:
        fig.add_trace(
          go.Bar(
            x=pivot.index.astype(str),
            y=pivot[stage],
            name=stage,
            marker_color=colors.get(stage, HEALTH_COLORS["neutral"]),
            hovertemplate=f"%{{x}}<br>{stage}: %{{y:.1f}} h<extra></extra>",
          )
        )

      fig.update_layout(
        barmode="stack",
        title=chart_title,
        xaxis_title=self.translator.t("chart.label.date"),
        yaxis_title=self.translator.t("chart.label.sleep_duration_hours"),
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.sleep_stages_distribution"),
          error=str(e),
        )
      )
      return None

  def plot_sleep_consistency(
    self,
    data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot sleep consistency chart.

    Args:
        data: Sleep data (columns: date, bedtime, wake_time).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if data.empty:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.sleep_consistency"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating",
        chart=self.translator.t("chart.title.sleep_consistency"),
        count=len(data),
      )
    )
    chart_title = title or self.translator.t("chart.title.sleep_consistency")

    try:
      fig = go.Figure()

      # Bedtime.
      if "bedtime" in data.columns:
        # Convert time to hours (24-hour format).
        bedtime_hours = data["bedtime"].apply(
          lambda x: x.hour + x.minute / 60 if pd.notna(x) else None
        )

        fig.add_trace(
          go.Scatter(
            x=data["date"],
            y=bedtime_hours,
            mode="markers+lines",
            name=self.translator.t("chart.label.bedtime"),
            marker={"size": 6, "color": HEALTH_COLORS["primary"]},
            line={"color": HEALTH_COLORS["primary"], "width": 1.5},
          )
        )

      # Wake time.
      if "wake_time" in data.columns:
        wake_hours = data["wake_time"].apply(
          lambda x: x.hour + x.minute / 60 if pd.notna(x) else None
        )

        fig.add_trace(
          go.Scatter(
            x=data["date"],
            y=wake_hours,
            mode="markers+lines",
            name=self.translator.t("chart.label.wake_time"),
            marker={"size": 6, "color": HEALTH_COLORS["secondary"]},
            line={"color": HEALTH_COLORS["secondary"], "width": 1.5},
          )
        )

      fig.update_layout(
        title=chart_title,
        xaxis_title=self.translator.t("chart.label.date"),
        yaxis_title=self.translator.t("chart.label.time_24h"),
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.sleep_consistency"),
          error=str(e),
        )
      )
      return None

  def plot_weekday_vs_weekend_sleep(
    self,
    data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot weekday vs weekend sleep comparison.

    Args:
        data: Sleep data (columns: date, duration, is_weekend).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if data.empty:
      logger.warning(
        self.translator.t(
          "log.chart_no_data",
          chart=self.translator.t("chart.title.weekday_vs_weekend_sleep"),
        )
      )
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating_generic",
        chart=self.translator.t("chart.title.weekday_vs_weekend_sleep"),
      )
    )
    chart_title = title or self.translator.t("chart.title.weekday_vs_weekend_sleep")

    try:
      # Group data.
      weekday_data = data[~data["is_weekend"]]["duration"]
      weekend_data = data[data["is_weekend"]]["duration"]

      fig = go.Figure()

      # Weekday box plot.
      fig.add_trace(
        go.Box(
          y=weekday_data,
          name=self.translator.t("chart.label.weekday"),
          marker_color=HEALTH_COLORS["primary"],
          boxmean="sd",
        )
      )

      # Weekend box plot.
      fig.add_trace(
        go.Box(
          y=weekend_data,
          name=self.translator.t("chart.label.weekend"),
          marker_color=HEALTH_COLORS["secondary"],
          boxmean="sd",
        )
      )

      fig.update_layout(
        title=chart_title,
        yaxis_title=self.translator.t("chart.label.sleep_duration_hours"),
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.weekday_vs_weekend_sleep"),
          error=str(e),
        )
      )
      return None

  def generate_heart_rate_report_charts(
    self,
    report: HeartRateAnalysisReport,
    output_dir: Path,
  ) -> dict[str, Path]:
    """Generate all charts for a heart rate report.

    Args:
        report: Heart rate analysis report.
        output_dir: Output directory.

    Returns:
        Mapping of chart names to file paths.
    """
    logger.info("Generating heart rate report charts")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts: dict[str, Path] = {}

    try:
      # Resting heart rate trend chart.
      if (
        report.resting_hr_analysis
        and report.daily_stats is not None
        and not report.daily_stats.empty
      ):
        resting_hr_path = output_dir / "resting_hr_trend.html"
        fig = self.plot_resting_hr_trend(
          report.daily_stats, output_path=resting_hr_path
        )
        if fig:
          charts["resting_hr_trend"] = resting_hr_path

      # HRV analysis chart.
      if (
        report.hrv_analysis
        and report.daily_stats is not None
        and not report.daily_stats.empty
      ):
        hrv_path = output_dir / "hrv_analysis.html"
        fig = self.plot_hrv_analysis(report.daily_stats, output_path=hrv_path)
        if fig:
          charts["hrv_analysis"] = hrv_path

      # Heart rate heatmap.
      if report.daily_stats is not None and not report.daily_stats.empty:
        heatmap_path = output_dir / "heart_rate_heatmap.html"
        # Prepare heatmap data.
        heatmap_data = report.daily_stats.copy()
        heatmap_data["date"] = pd.to_datetime(heatmap_data["interval_start"])
        heatmap_data["avg_hr"] = heatmap_data["mean_value"]
        fig = self.plot_heart_rate_heatmap(heatmap_data, output_path=heatmap_path)
        if fig:
          charts["heart_rate_heatmap"] = heatmap_path

      # Heart rate distribution.
      if report.daily_stats is not None and not report.daily_stats.empty:
        distribution_path = output_dir / "heart_rate_distribution.html"
        fig = self.plot_heart_rate_distribution(
          report.daily_stats, output_path=distribution_path
        )
        if fig:
          charts["heart_rate_distribution"] = distribution_path

      # Heart rate zones.
      if report.daily_stats is not None and not report.daily_stats.empty:
        zones_path = output_dir / "heart_rate_zones.html"
        fig = self.plot_heart_rate_zones(report.daily_stats, output_path=zones_path)
        if fig:
          charts["heart_rate_zones"] = zones_path

      # Advanced heart rate metrics.
      if report.advanced_metrics:
        advanced_path = output_dir / "heart_rate_advanced.html"
        fig = self.plot_heart_rate_advanced_metrics(report, output_path=advanced_path)
        if fig:
          charts["heart_rate_advanced"] = advanced_path

      logger.info(
        self.translator.t(
          "log.charts.generated_heart_rate",
          count=len(charts),
        )
      )

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.charts.generate_heart_rate_failed",
          error=e,
        )
      )

    return charts

  def generate_sleep_report_charts(
    self,
    report: SleepAnalysisReport,
    output_dir: Path,
  ) -> dict[str, Path]:
    """Generate all charts for a sleep report.

    Args:
        report: Sleep analysis report.
        output_dir: Output directory.

    Returns:
        Mapping of chart names to file paths.
    """
    logger.info(self.translator.t("log.charts.generating_sleep"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts: dict[str, Path] = {}

    try:
      # Sleep timeline chart.
      if report.sleep_sessions:
        timeline_path = output_dir / "sleep_timeline.html"
        # Prepare timeline data.
        timeline_data = []
        for session in report.sleep_sessions:
          timeline_data.append(
            {
              "start_date": session.start_date,
              "end_date": session.end_date,
              "value": "Asleep",  # Simplified handling.
            }
          )
        timeline_df = pd.DataFrame(timeline_data)
        fig = self.plot_sleep_timeline(timeline_df, output_path=timeline_path)
        if fig:
          charts["sleep_timeline"] = timeline_path

      # Sleep quality trend chart.
      if report.daily_summary is not None and not report.daily_summary.empty:
        quality_trend_path = output_dir / "sleep_quality_trend.html"
        fig = self.plot_sleep_quality_trend(
          report.daily_summary, output_path=quality_trend_path
        )
        if fig:
          charts["sleep_quality_trend"] = quality_trend_path

        details_path = output_dir / "sleep_quality_details.html"
        fig = self.plot_sleep_quality_details(
          report.daily_summary, output_path=details_path
        )
        if fig:
          charts["sleep_quality_details"] = details_path

      # Sleep stage distribution.
      if report.daily_summary is not None and not report.daily_summary.empty:
        stages_path = output_dir / "sleep_stages_distribution.html"
        # Prepare stage data.
        stages_data = []
        if "date" not in report.daily_summary.columns:
          logger.warning("sleep stages: daily_summary missing date column")
        else:
          logger.info(
            "sleep stages: rows=%s cols=%s",
            len(report.daily_summary),
            list(report.daily_summary.columns),
          )
          logger.info(
            "sleep stages: non-null counts deep=%s rem=%s light=%s sleep=%s total=%s",
            report.daily_summary.get("deep_sleep", pd.Series(dtype=float))
            .notna()
            .sum(),
            report.daily_summary.get("rem_sleep", pd.Series(dtype=float)).notna().sum(),
            report.daily_summary.get("light_sleep", pd.Series(dtype=float))
            .notna()
            .sum(),
            report.daily_summary.get("sleep_duration", pd.Series(dtype=float))
            .notna()
            .sum(),
            report.daily_summary.get("total_duration", pd.Series(dtype=float))
            .notna()
            .sum(),
          )
        for _, row in report.daily_summary.iterrows():
          date_val = row.get("date")
          if pd.isna(date_val):
            continue
          deep_val = float(row.get("deep_sleep", 0) or 0)
          rem_val = float(row.get("rem_sleep", 0) or 0)
          total_sleep = float(row.get("sleep_duration", 0) or 0)
          if total_sleep <= 0:
            total_sleep = float(row.get("total_duration", 0) or 0)
          light_val = float(row.get("light_sleep", 0) or 0)
          if light_val <= 0:
            light_val = max(0.0, total_sleep - deep_val - rem_val)
          row_added = False
          if deep_val > 0:
            stages_data.append(
              {"date": date_val, "stage": "Deep", "duration": deep_val / 60}
            )
            row_added = True
          if rem_val > 0:
            stages_data.append(
              {"date": date_val, "stage": "REM", "duration": rem_val / 60}
            )
            row_added = True
          if light_val > 0:
            stages_data.append(
              {"date": date_val, "stage": "Light", "duration": light_val / 60}
            )
            row_added = True
          if not row_added and total_sleep > 0:
            stages_data.append(
              {"date": date_val, "stage": "Light", "duration": total_sleep / 60}
            )
        if stages_data:
          stages_df = pd.DataFrame(stages_data)
          fig = self.plot_sleep_stages_distribution(stages_df, output_path=stages_path)
          if fig:
            charts["sleep_stages_distribution"] = stages_path

          max_date = pd.to_datetime(stages_df["date"]).max()
          if pd.notna(max_date):
            last_7_start = max_date - pd.Timedelta(days=6)
            last_7_df = stages_df[pd.to_datetime(stages_df["date"]) >= last_7_start]
            if not last_7_df.empty:
              stages_7d_path = output_dir / "sleep_stages_distribution_7d.html"
              fig = self.plot_sleep_stages_distribution(
                last_7_df,
                title=self.translator.t("chart.title.sleep_stages_distribution_7d"),
                output_path=stages_7d_path,
              )
              if fig:
                charts["sleep_stages_distribution_7d"] = stages_7d_path

            weekly_df = stages_df.copy()
            weekly_df["date"] = (
              pd.to_datetime(weekly_df["date"]).dt.to_period("W").dt.start_time
            )
            stages_weekly_path = output_dir / "sleep_stages_distribution_weekly.html"
            fig = self.plot_sleep_stages_distribution(
              weekly_df,
              title=self.translator.t("chart.title.sleep_stages_distribution_weekly"),
              output_path=stages_weekly_path,
            )
            if fig:
              charts["sleep_stages_distribution_weekly"] = stages_weekly_path

      # Sleep consistency analysis.
      if report.daily_summary is not None and not report.daily_summary.empty:
        consistency_path = output_dir / "sleep_consistency.html"
        # Prepare consistency data.
        consistency_data = []
        for _, row in report.daily_summary.iterrows():
          consistency_data.append(
            {
              "date": row["date"],
              "bedtime": None,  # Needs extraction from session data.
              "wake_time": None,
            }
          )
        consistency_df = pd.DataFrame(consistency_data)
        fig = self.plot_sleep_consistency(consistency_df, output_path=consistency_path)
        if fig:
          charts["sleep_consistency"] = consistency_path

      # Weekday vs weekend sleep comparison.
      if report.daily_summary is not None and not report.daily_summary.empty:
        weekday_weekend_path = output_dir / "weekday_vs_weekend_sleep.html"
        # Prepare comparison data.
        comparison_data = []
        for _, row in report.daily_summary.iterrows():
          date = pd.to_datetime(row["date"])
          is_weekend = date.weekday() >= 5
          comparison_data.append(
            {
              "date": row["date"],
              "duration": row["total_duration"] / 60,  # Convert to hours.
              "is_weekend": is_weekend,
            }
          )
        comparison_df = pd.DataFrame(comparison_data)
        fig = self.plot_weekday_vs_weekend_sleep(
          comparison_df, output_path=weekday_weekend_path
        )
        if fig:
          charts["weekday_vs_weekend_sleep"] = weekday_weekend_path

      logger.info(
        self.translator.t(
          "log.charts.generated_sleep",
          count=len(charts),
        )
      )

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.charts.generate_sleep_failed",
          error=e,
        )
      )

    return charts

  def plot_health_dashboard(
    self,
    wellness_score: float,
    metrics: dict[str, float],
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot health dashboard chart.

    Args:
        wellness_score: Overall wellness score (0-1).
        metrics: Health metrics dictionary.
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    logger.info(
      self.translator.t(
        "log.chart_generating_generic",
        chart=self.translator.t("chart.title.health_dashboard"),
      )
    )
    chart_title = title or self.translator.t("chart.title.health_dashboard")

    try:
      # Create subplot layout.
      fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
          self.translator.t("chart.subtitle.overall_wellness"),
          self.translator.t("chart.subtitle.key_metrics_radar"),
          self.translator.t("chart.subtitle.metric_trends"),
          self.translator.t("chart.subtitle.health_distribution"),
        ),
        specs=[
          [{"type": "indicator"}, {"type": "scatterpolar"}],
          [{"type": "bar"}, {"type": "pie"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
      )

      # 1. Overall wellness gauge.
      fig.add_trace(
        go.Indicator(
          mode="gauge+number",
          value=wellness_score * 100,
          title={"text": self.translator.t("chart.label.health_score")},
          gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": HEALTH_COLORS["primary"]},
            "steps": [
              {"range": [0, 40], "color": HEALTH_COLORS["danger"]},
              {"range": [40, 70], "color": HEALTH_COLORS["warning"]},
              {"range": [70, 100], "color": HEALTH_COLORS["success"]},
            ],
            "threshold": {
              "line": {"color": "black", "width": 4},
              "thickness": 0.75,
              "value": wellness_score * 100,
            },
          },
        ),
        row=1,
        col=1,
      )

      # 2. Key metrics radar chart.
      if metrics:
        categories = list(metrics.keys())
        values = [metrics[cat] for cat in categories]

        fig.add_trace(
          go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=self.translator.t("chart.title.circular_metrics"),
            line_color=HEALTH_COLORS["secondary"],
            fillcolor=HEALTH_COLORS["secondary"],
            opacity=0.3,
          ),
          row=1,
          col=2,
        )

        fig.update_polars(
          radialaxis={"range": [0, 1], "showticklabels": False},
          row=1,
          col=2,
        )

      # 3. Metric trend bar chart.
      if metrics:
        fig.add_trace(
          go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=HEALTH_COLORS["info"],
            name=self.translator.t("chart.label.metric_value"),
          ),
          row=2,
          col=1,
        )

      # 4. Health distribution pie chart.
      health_categories = {
        self.translator.t("chart.health.excellent"): (max(0, wellness_score - 0.8) * 5),
        self.translator.t("chart.health.good"): (
          max(0, min(0.8, wellness_score) - 0.6) * 5
        ),
        self.translator.t("chart.health.fair"): (
          max(0, min(0.6, wellness_score) - 0.4) * 5
        ),
        self.translator.t("chart.health.attention"): (max(0, 0.4 - wellness_score) * 5),
      }

      fig.add_trace(
        go.Pie(
          labels=list(health_categories.keys()),
          values=list(health_categories.values()),
          marker_colors=[
            HEALTH_COLORS["success"],
            HEALTH_COLORS["primary"],
            HEALTH_COLORS["warning"],
            HEALTH_COLORS["danger"],
          ],
          name=self.translator.t("chart.subtitle.health_distribution"),
        ),
        row=2,
        col=2,
      )

      fig.update_layout(
        title=chart_title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        showlegend=False,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.health_dashboard"),
          error=str(e),
        )
      )
      return None

  def plot_correlation_heatmap(
    self,
    correlation_data: dict[str, Any],
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot health metric correlation heatmap.

    Args:
        correlation_data: Correlation data dictionary.
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    logger.info(
      self.translator.t(
        "log.chart_generating_generic",
        chart=self.translator.t("chart.title.correlation_heatmap"),
      )
    )
    chart_title = title or self.translator.t("chart.title.correlation_heatmap")

    try:
      # Extract correlation matrix.
      correlations = {}
      for key, data in correlation_data.items():
        if isinstance(data, dict) and "correlation" in data:
          correlations[key] = data["correlation"]

      if not correlations:
        logger.warning(
          self.translator.t(
            "log.chart_no_data",
            chart=self.translator.t("chart.title.correlation_heatmap"),
          )
        )
        return None

      # Build correlation matrix.
      metrics = list(correlations.keys())
      matrix = np.eye(len(metrics))  # Diagonal is 1.

      # Fill correlation values.
      for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
          if i != j:
            key = f"{metric1}_{metric2}"
            rev_key = f"{metric2}_{metric1}"
            if key in correlations:
              matrix[i, j] = correlations[key]
            elif rev_key in correlations:
              matrix[i, j] = correlations[rev_key]

      fig = go.Figure(
        data=go.Heatmap(
          z=matrix,
          x=metrics,
          y=metrics,
          colorscale="RdBu",
          zmid=0,
          colorbar={"title": self.translator.t("chart.label.correlation")},
          hovertemplate="<b>%{x} vs %{y}</b><br>"
          + f"{self.translator.t('chart.label.metric_value')}: %{{z:.2f}}<br>"
          + "<extra></extra>",
        )
      )

      fig.update_layout(
        title=chart_title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.correlation_heatmap"),
          error=str(e),
        )
      )
      return None

  def plot_trend_analysis(
    self,
    trend_data: dict[str, list[float]],
    dates: list[str],
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot health trend analysis chart.

    Args:
        trend_data: Trend data dictionary.
        dates: Date labels.
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    logger.info(
      self.translator.t(
        "log.chart_generating_generic",
        chart=self.translator.t("chart.title.trend_analysis"),
      )
    )
    chart_title = title or self.translator.t("chart.title.trend_analysis")

    try:
      fig = go.Figure()

      colors = [
        HEALTH_COLORS["primary"],
        HEALTH_COLORS["secondary"],
        HEALTH_COLORS["success"],
        HEALTH_COLORS["info"],
        HEALTH_COLORS["warning"],
      ]

      for i, (metric_name, values) in enumerate(trend_data.items()):
        color = colors[i % len(colors)]

        fig.add_trace(
          go.Scatter(
            x=dates,
            y=values,
            mode="lines+markers",
            name=metric_name,
            line={"color": color, "width": 2},
            marker={"size": 6, "color": color},
          )
        )

      fig.update_layout(
        title=chart_title,
        xaxis_title=self.translator.t("chart.label.date"),
        yaxis_title=self.translator.t("chart.label.metric_value"),
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.trend_analysis"),
          error=str(e),
        )
      )
      return None

  def plot_activity_heatmap(
    self,
    activity_data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot activity pattern heatmap.

    Args:
        activity_data: Activity data (columns: date, hour, activity_level).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if activity_data.empty:
      logger.warning("No activity data for heatmap")
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating_generic",
        chart=self.translator.t("chart.title.activity_heatmap"),
      )
    )
    chart_title = title or self.translator.t("chart.title.activity_heatmap")

    try:
      # Prepare heatmap data.
      activity_data = activity_data.copy()
      activity_data["date"] = pd.to_datetime(activity_data["date"])
      activity_data["weekday"] = activity_data["date"].dt.dayofweek
      activity_data["week"] = activity_data["date"].dt.isocalendar().week

      # Create pivot table: weekday x hour.
      pivot = activity_data.pivot_table(
        values="activity_level",
        index="weekday",
        columns="hour",
        aggfunc="mean",
      )

      weekdays = [
        self.translator.t("chart.weekday.mon"),
        self.translator.t("chart.weekday.tue"),
        self.translator.t("chart.weekday.wed"),
        self.translator.t("chart.weekday.thu"),
        self.translator.t("chart.weekday.fri"),
        self.translator.t("chart.weekday.sat"),
        self.translator.t("chart.weekday.sun"),
      ]
      hours = [f"{h:02d}:00" for h in range(24)]

      fig = go.Figure(
        data=go.Heatmap(
          z=pivot.values,
          x=hours,
          y=weekdays,
          colorscale="Viridis",
          colorbar={"title": self.translator.t("chart.label.activity_level")},
          hovertemplate="<b>%{y} %{x}</b><br>"
          + f"{self.translator.t('chart.label.activity_level')}: %{{z:.2f}}<br>"
          + "<extra></extra>",
        )
      )

      fig.update_layout(
        title=chart_title,
        xaxis_title=self.translator.t("chart.label.hour"),
        yaxis_title=self.translator.t("chart.label.weekday"),
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.activity_heatmap"),
          error=str(e),
        )
      )
      return None

  def plot_circular_health_metrics(
    self,
    metrics: dict[str, float],
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot circular health metrics chart.

    Args:
        metrics: Health metrics dictionary.
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    logger.info(
      self.translator.t(
        "log.chart_generating_generic",
        chart=self.translator.t("chart.title.circular_metrics"),
      )
    )
    chart_title = title or self.translator.t("chart.title.circular_metrics")

    try:
      if not metrics:
        return None

      # Create circular bar chart.
      categories = list(metrics.keys())
      values = list(metrics.values())

      fig = go.Figure()

      # Background ring.
      fig.add_trace(
        go.Barpolar(
          r=[1] * len(categories),
          theta=categories,
          marker_color="lightgray",
          opacity=0.3,
          showlegend=False,
        )
      )

      # Data rings.
      colors = [
        HEALTH_COLORS["primary"],
        HEALTH_COLORS["secondary"],
        HEALTH_COLORS["success"],
        HEALTH_COLORS["info"],
        HEALTH_COLORS["warning"],
        HEALTH_COLORS["danger"],
      ]

      fig.add_trace(
        go.Barpolar(
          r=values,
          theta=categories,
          marker_color=[colors[i % len(colors)] for i in range(len(categories))],
          name=self.translator.t("chart.title.circular_metrics"),
          hovertemplate="<b>%{theta}</b><br>"
          + f"{self.translator.t('chart.label.metric_value')}: %{{r:.2f}}<br>"
          + "<extra></extra>",
        )
      )

      fig.update_layout(
        title=chart_title,
        polar={
          "radialaxis": {"range": [0, 1], "showticklabels": False},
          "angularaxis": {"showticklabels": True},
        },
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.circular_metrics"),
          error=str(e),
        )
      )
      return None

  def plot_health_timeline(
    self,
    timeline_data: pd.DataFrame,
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot health timeline chart.

    Args:
        timeline_data: Timeline data (columns: date, metric, value, category).
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    if timeline_data.empty:
      logger.warning("No timeline data for health timeline")
      return None

    logger.info(
      self.translator.t(
        "log.chart_generating_generic",
        chart=self.translator.t("chart.title.health_timeline"),
      )
    )
    chart_title = title or self.translator.t("chart.title.health_timeline")

    try:
      fig = go.Figure()

      # Plot by category.
      categories = timeline_data["category"].unique()
      colors = [
        HEALTH_COLORS["primary"],
        HEALTH_COLORS["secondary"],
        HEALTH_COLORS["success"],
        HEALTH_COLORS["info"],
        HEALTH_COLORS["warning"],
      ]

      for i, category in enumerate(categories):
        cat_data = timeline_data[timeline_data["category"] == category]
        color = colors[i % len(colors)]

        fig.add_trace(
          go.Scatter(
            x=cat_data["date"],
            y=cat_data["value"],
            mode="lines+markers",
            name=category,
            line={"color": color, "width": 2},
            marker={"size": 6, "color": color},
            hovertemplate="<b>%{fullData.name}</b><br>"
            + f"{self.translator.t('chart.label.date')}: %{{x}}<br>"
            + f"{self.translator.t('chart.label.metric_value')}: %{{y:.2f}}<br>"
            + "<extra></extra>",
          )
        )

      fig.update_layout(
        title=chart_title,
        xaxis_title=self.translator.t("chart.label.date"),
        yaxis_title=self.translator.t("chart.label.metric_value"),
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.health_timeline"),
          error=str(e),
        )
      )
      return None

  def plot_risk_assessment(
    self,
    risk_factors: dict[str, float],
    title: str | None = None,
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """Plot health risk assessment chart.

    Args:
        risk_factors: Risk factors dictionary.
        title: Chart title.
        output_path: Output path.

    Returns:
        Plotly Figure or None.
    """
    logger.info(
      self.translator.t(
        "log.chart_generating_generic",
        chart=self.translator.t("chart.title.risk_assessment"),
      )
    )
    chart_title = title or self.translator.t("chart.title.risk_assessment")

    try:
      if not risk_factors:
        return None

      # Build a bar chart of risk factors.
      factors = list(risk_factors.keys())
      values = list(risk_factors.values())

      fig = go.Figure()

      # Color bars by risk level.
      colors = []
      for value in values:
        if value > 0.7:
          colors.append(HEALTH_COLORS["danger"])
        elif value > 0.4:
          colors.append(HEALTH_COLORS["warning"])
        else:
          colors.append(HEALTH_COLORS["success"])

      fig.add_trace(
        go.Bar(
          x=factors,
          y=values,
          marker_color=colors,
          name=self.translator.t("chart.title.risk_assessment"),
          hovertemplate="<b>%{x}</b><br>"
          + f"{self.translator.t('chart.label.risk_level')}: %{{y:.2f}}<br>"
          + "<extra></extra>",
        )
      )

      # Add risk threshold lines.
      fig.add_hline(
        y=0.7,
        line_dash="dash",
        line_color=HEALTH_COLORS["danger"],
        annotation_text=self.translator.t("chart.label.risk_high"),
        annotation_position="right",
      )

      fig.add_hline(
        y=0.4,
        line_dash="dash",
        line_color=HEALTH_COLORS["warning"],
        annotation_text=self.translator.t("chart.label.risk_medium"),
        annotation_position="right",
      )

      fig.update_layout(
        title=chart_title,
        xaxis_title=self.translator.t("chart.label.risk_factor"),
        yaxis_title=self.translator.t("chart.label.risk_level"),
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.chart_error",
          chart=self.translator.t("chart.title.risk_assessment"),
          error=str(e),
        )
      )
      return None

  def generate_comprehensive_report_charts(
    self,
    report: Any,  # ComprehensiveHealthReport
    output_dir: Path,
  ) -> dict[str, Path]:
    """Generate all charts for a comprehensive report.

    Args:
        report: Comprehensive health analysis report.
        output_dir: Output directory.

    Returns:
        Mapping of chart names to file paths.
    """
    logger.info("Generating comprehensive health report charts")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = {}

    try:
      # Health dashboard.
      if hasattr(report, "overall_wellness_score"):
        dashboard_path = output_dir / "health_dashboard.html"
        metrics = {}

        # Collect metrics.
        if hasattr(report, "sleep_quality") and report.sleep_quality:
          metrics[self.translator.t("chart.metric.sleep_quality")] = min(
            1.0, report.sleep_quality.average_duration_hours / 8.0
          )
        if hasattr(report, "activity_patterns") and report.activity_patterns:
          metrics[self.translator.t("chart.metric.activity_level")] = min(
            1.0, report.activity_patterns.daily_step_average / 10000
          )
        if hasattr(report, "metabolic_health") and report.metabolic_health:
          metrics[self.translator.t("chart.metric.metabolic_health")] = (
            report.metabolic_health.metabolic_health_score
          )
        if hasattr(report, "stress_resilience") and report.stress_resilience:
          metrics[self.translator.t("chart.metric.stress_resilience")] = (
            1.0 - report.stress_resilience.stress_accumulation_score
          )

        dashboard_fig = self.plot_health_dashboard(
          report.overall_wellness_score,
          metrics,
          output_path=dashboard_path,
        )
        if dashboard_fig:
          charts["dashboard"] = dashboard_path

      # Correlation heatmap.
      if hasattr(report, "health_correlations") and report.health_correlations:
        correlation_path = output_dir / "correlation_heatmap.html"
        correlation_fig = self.plot_correlation_heatmap(
          report.health_correlations,
          output_path=correlation_path,
        )
        if correlation_fig:
          charts["correlation"] = correlation_path

      # Health timeline.
      # Timeline data preparation is pending; skip for now.

      # Risk assessment.
      if hasattr(report, "stress_resilience") and report.stress_resilience:
        risk_path = output_dir / "risk_assessment.html"
        risk_factors = {
          self.translator.t("chart.risk.stress_accumulation"): (
            report.stress_resilience.stress_accumulation_score
          ),
          self.translator.t("chart.risk.recovery_capacity"): (
            1.0 - report.stress_resilience.recovery_capacity_score
          ),
        }

        risk_fig = self.plot_risk_assessment(
          risk_factors,
          output_path=risk_path,
        )
        if risk_fig:
          charts["risk_assessment"] = risk_path

      logger.info(
        self.translator.t(
          "log.charts.generated_comprehensive",
          count=len(charts),
        )
      )

    except Exception as e:
      logger.error(
        self.translator.t(
          "log.charts.generate_comprehensive_failed",
          error=e,
        )
      )

    return charts
