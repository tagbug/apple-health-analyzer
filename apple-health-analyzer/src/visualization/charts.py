"""图表生成模块 - 提供各类健康数据可视化图表"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from ..processors.heart_rate import HeartRateAnalysisReport
from ..processors.sleep import SleepAnalysisReport
from ..utils.logger import get_logger

logger = get_logger(__name__)

# 健康主题配色方案
HEALTH_COLORS = {
  "primary": "#4CAF50",  # 健康绿
  "secondary": "#2196F3",  # 信息蓝
  "warning": "#FF9800",  # 橙色
  "danger": "#F44336",  # 红色
  "neutral": "#9E9E9E",  # 灰色
  "success": "#8BC34A",  # 浅绿
  "info": "#03A9F4",  # 浅蓝
  "light": "#E8F5E9",  # 浅绿背景
  "dark": "#1B5E20",  # 深绿
}

# Plotly 主题配置
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
  """图表生成核心类

  提供各类健康数据的可视化图表生成功能，支持交互式和静态图表。
  """

  def __init__(
    self,
    theme: str = "health",
    width: int = 1200,
    height: int = 600,
    dpi: int = 300,
  ):
    """初始化图表生成器

    Args:
        theme: 主题名称 (health/light/dark)
        width: 图表宽度
        height: 图表高度
        dpi: 图表DPI (用于静态图导出)
    """
    self.theme = theme
    self.width = width
    self.height = height
    self.dpi = dpi

    # 设置 matplotlib 样式
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    logger.info(
      f"ChartGenerator initialized: theme={theme}, size={width}x{height}"
    )

  def plot_heart_rate_timeseries(
    self,
    data: pd.DataFrame,
    title: str = "心率时序图",
    output_path: Path | None = None,
    interactive: bool = True,
  ) -> go.Figure | None:
    """绘制心率时序图

    Args:
        data: 包含时间和心率数据的DataFrame (columns: timestamp, value)
        title: 图表标题
        output_path: 输出路径 (可选)
        interactive: 是否生成交互式图表

    Returns:
        Plotly Figure 对象或 None
    """
    if data.empty:
      logger.warning("No data provided for heart rate timeseries")
      return None

    logger.info(
      f"Generating heart rate timeseries chart with {len(data)} points"
    )

    # 数据采样 (如果数据量过大)
    if len(data) > 10000:
      logger.info(f"Downsampling from {len(data)} to 10000 points")
      data = self._downsample_data(data, 10000)

    try:
      if interactive:
        # 使用 Plotly 生成交互式图表
        fig = go.Figure()

        fig.add_trace(
          go.Scatter(
            x=data["timestamp"],
            y=data["value"],
            mode="lines",
            name="心率",
            line=dict(color=HEALTH_COLORS["primary"], width=1.5),
            hovertemplate="<b>时间</b>: %{x}<br>"
            + "<b>心率</b>: %{y:.0f} bpm<br>"
            + "<extra></extra>",
          )
        )

        # 添加平均线
        mean_hr = data["value"].mean()
        fig.add_hline(
          y=mean_hr,
          line_dash="dash",
          line_color=HEALTH_COLORS["secondary"],
          annotation_text=f"平均值: {mean_hr:.0f} bpm",
          annotation_position="right",
        )

        fig.update_layout(
          title=title,
          xaxis_title="时间",
          yaxis_title="心率 (bpm)",
          width=self.width,
          height=self.height,
          template=PLOTLY_TEMPLATE,
          hovermode="x unified",
        )

        if output_path:
          self._save_plotly_figure(fig, output_path)

        return fig
      else:
        # 使用 Matplotlib 生成静态图表
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
          label=f"平均值: {data['value'].mean():.0f} bpm",
        )
        ax.set_xlabel("时间")
        ax.set_ylabel("心率 (bpm)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_path:
          plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
          logger.info(f"Chart saved to {output_path}")

        plt.close()
        return None

    except Exception as e:
      logger.error(f"Error generating heart rate timeseries: {e}")
      return None

  def plot_resting_hr_trend(
    self,
    data: pd.DataFrame,
    title: str = "静息心率趋势",
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """绘制静息心率趋势图

    Args:
        data: 静息心率数据 (columns: start_date, value)
        title: 图表标题
        output_path: 输出路径

    Returns:
        Plotly Figure 对象或 None
    """
    if data.empty:
      logger.warning("No data for resting heart rate trend")
      return None

    logger.info(
      f"Generating resting heart rate trend chart with {len(data)} points"
    )

    try:
      fig = go.Figure()

      # 实际数据点
      fig.add_trace(
        go.Scatter(
          x=data["start_date"],
          y=data["value"],
          mode="markers+lines",
          name="静息心率",
          marker=dict(size=6, color=HEALTH_COLORS["primary"], opacity=0.7),
          line=dict(color=HEALTH_COLORS["primary"], width=2),
        )
      )

      # 添加趋势线 (移动平均)
      if len(data) > 7:
        ma_7 = data["value"].rolling(window=7, center=True).mean()
        fig.add_trace(
          go.Scatter(
            x=data["start_date"],
            y=ma_7,
            mode="lines",
            name="7天移动平均",
            line=dict(
              color=HEALTH_COLORS["secondary"],
              width=2,
              dash="dash",
            ),
          )
        )

      # 添加健康范围区域
      fig.add_hrect(
        y0=60,
        y1=100,
        fillcolor=HEALTH_COLORS["success"],
        opacity=0.1,
        layer="below",
        line_width=0,
        annotation_text="健康范围",
        annotation_position="right",
      )

      fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="静息心率 (bpm)",
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(f"Error generating resting heart rate trend: {e}")
      return None

  def plot_hrv_analysis(
    self,
    data: pd.DataFrame,
    title: str = "心率变异性 (HRV) 分析",
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """绘制HRV分析图

    Args:
        data: HRV数据 (columns: start_date, value)
        title: 图表标题
        output_path: 输出路径

    Returns:
        Plotly Figure 对象或 None
    """
    if data.empty:
      logger.warning("No data for HRV analysis")
      return None

    logger.info(f"Generating HRV analysis chart with {len(data)} points")

    try:
      # 创建子图: HRV趋势 + 分布直方图
      fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("HRV时序趋势", "HRV分布"),
        vertical_spacing=0.15,
      )

      # HRV时序图
      fig.add_trace(
        go.Scatter(
          x=data["start_date"],
          y=data["value"],
          mode="lines+markers",
          name="SDNN",
          line=dict(color=HEALTH_COLORS["info"], width=2),
          marker=dict(size=4),
        ),
        row=1,
        col=1,
      )

      # 添加移动平均
      if len(data) > 7:
        ma = data["value"].rolling(window=7, center=True).mean()
        fig.add_trace(
          go.Scatter(
            x=data["start_date"],
            y=ma,
            mode="lines",
            name="7天平均",
            line=dict(
              color=HEALTH_COLORS["secondary"],
              width=2,
              dash="dash",
            ),
          ),
          row=1,
          col=1,
        )

      # HRV分布直方图
      fig.add_trace(
        go.Histogram(
          x=data["value"],
          name="分布",
          marker_color=HEALTH_COLORS["info"],
          opacity=0.7,
          nbinsx=30,
        ),
        row=2,
        col=1,
      )

      # 更新布局
      fig.update_xaxes(title_text="日期", row=1, col=1)
      fig.update_yaxes(title_text="SDNN (ms)", row=1, col=1)
      fig.update_xaxes(title_text="SDNN (ms)", row=2, col=1)
      fig.update_yaxes(title_text="频数", row=2, col=1)

      fig.update_layout(
        title=title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        showlegend=True,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(f"Error generating HRV analysis: {e}")
      return None

  def plot_heart_rate_heatmap(
    self,
    data: pd.DataFrame,
    title: str = "心率热力图 (日历视图)",
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """绘制心率热力图 (日历视图)

    Args:
        data: 每日心率数据 (columns: date, avg_hr)
        title: 图表标题
        output_path: 输出路径

    Returns:
        Plotly Figure 对象或 None
    """
    if data.empty:
      logger.warning("No data for heart rate heatmap")
      return None

    logger.info(f"Generating heart rate heatmap with {len(data)} days")

    try:
      # 准备热力图数据: 周 x 星期几
      data = data.copy()
      data["date"] = pd.to_datetime(data["date"])
      data["week"] = data["date"].dt.isocalendar().week
      data["weekday"] = data["date"].dt.dayofweek
      data["year"] = data["date"].dt.year

      # 创建透视表
      pivot = data.pivot_table(
        values="avg_hr", index="week", columns="weekday", aggfunc="mean"
      )

      # 创建热力图
      fig = go.Figure(
        data=go.Heatmap(
          z=pivot.values,
          x=["周一", "周二", "周三", "周四", "周五", "周六", "周日"],
          y=pivot.index,
          colorscale="RdYlGn_r",  # 红-黄-绿（反向）
          colorbar=dict(title="平均心率<br>(bpm)"),
          hovertemplate="<b>第%{y}周 %{x}</b><br>"
          + "平均心率: %{z:.0f} bpm<br>"
          + "<extra></extra>",
        )
      )

      fig.update_layout(
        title=title,
        xaxis_title="星期",
        yaxis_title="周数",
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(f"Error generating heart rate heatmap: {e}")
      return None

  def plot_heart_rate_distribution(
    self,
    data: pd.DataFrame,
    title: str = "心率分布分析",
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """绘制心率分布直方图和箱线图

    Args:
        data: 心率数据 (columns: value)
        title: 图表标题
        output_path: 输出路径

    Returns:
        Plotly Figure 对象或 None
    """
    if data.empty or "value" not in data.columns:
      logger.warning("No data for heart rate distribution")
      return None

    logger.info(
      f"Generating heart rate distribution chart with {len(data)} points"
    )

    try:
      # 创建子图: 直方图 + 箱线图
      fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=("心率分布直方图", "箱线图"),
        horizontal_spacing=0.1,
      )

      # 直方图
      fig.add_trace(
        go.Histogram(
          x=data["value"],
          name="心率分布",
          marker_color=HEALTH_COLORS["primary"],
          opacity=0.7,
          nbinsx=50,
          hovertemplate="心率范围: %{x}<br>频数: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
      )

      # 添加正态分布曲线
      mean = data["value"].mean()
      std = data["value"].std()
      x_range = np.linspace(data["value"].min(), data["value"].max(), 100)
      normal_dist = (
        len(data)
        * (data["value"].max() - data["value"].min())
        / 50
        * (1 / (std * np.sqrt(2 * np.pi)))
        * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
      )

      fig.add_trace(
        go.Scatter(
          x=x_range,
          y=normal_dist,
          mode="lines",
          name="正态分布",
          line=dict(color=HEALTH_COLORS["secondary"], width=2, dash="dash"),
        ),
        row=1,
        col=1,
      )

      # 箱线图
      fig.add_trace(
        go.Box(
          y=data["value"],
          name="心率",
          marker_color=HEALTH_COLORS["primary"],
          boxmean="sd",  # 显示均值和标准差
        ),
        row=1,
        col=2,
      )

      # 更新布局
      fig.update_xaxes(title_text="心率 (bpm)", row=1, col=1)
      fig.update_yaxes(title_text="频数", row=1, col=1)
      fig.update_yaxes(title_text="心率 (bpm)", row=1, col=2)

      fig.update_layout(
        title=title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        showlegend=True,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(f"Error generating heart rate distribution: {e}")
      return None

  def plot_heart_rate_zones(
    self,
    data: pd.DataFrame,
    max_hr: float = 220,
    age: int = 30,
    title: str = "心率区间分布",
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """绘制心率区间分布图

    Args:
        data: 心率数据 (columns: value)
        max_hr: 最大心率 (默认: 220)
        age: 年龄 (用于计算最大心率)
        title: 图表标题
        output_path: 输出路径

    Returns:
        Plotly Figure 对象或 None
    """
    if data.empty or "value" not in data.columns:
      logger.warning("No data for heart rate zones")
      return None

    logger.info(f"Generating heart rate zones chart with {len(data)} points")

    try:
      # 计算年龄调整后的最大心率
      max_hr_adjusted = max_hr - age

      # 定义心率区间
      zones = {
        "休息区 (50-60%)": (
          0.5 * max_hr_adjusted,
          0.6 * max_hr_adjusted,
        ),
        "燃脂区 (60-70%)": (
          0.6 * max_hr_adjusted,
          0.7 * max_hr_adjusted,
        ),
        "有氧区 (70-80%)": (
          0.7 * max_hr_adjusted,
          0.8 * max_hr_adjusted,
        ),
        "无氧区 (80-90%)": (
          0.8 * max_hr_adjusted,
          0.9 * max_hr_adjusted,
        ),
        "极限区 (90-100%)": (
          0.9 * max_hr_adjusted,
          max_hr_adjusted,
        ),
      }

      # 统计每个区间的数据量
      zone_counts = {}
      for zone_name, (lower, upper) in zones.items():
        count = len(data[(data["value"] >= lower) & (data["value"] < upper)])
        zone_counts[zone_name] = count

      # 创建饼图
      fig = go.Figure(
        data=[
          go.Pie(
            labels=list(zone_counts.keys()),
            values=list(zone_counts.values()),
            marker=dict(
              colors=[
                HEALTH_COLORS["success"],
                HEALTH_COLORS["primary"],
                HEALTH_COLORS["info"],
                HEALTH_COLORS["warning"],
                HEALTH_COLORS["danger"],
              ]
            ),
            hovertemplate="<b>%{label}</b><br>"
            + "记录数: %{value}<br>"
            + "占比: %{percent}<br>"
            + "<extra></extra>",
          )
        ]
      )

      fig.update_layout(
        title=title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(f"Error generating heart rate zones: {e}")
      return None

  def _downsample_data(
    self, data: pd.DataFrame, target_points: int
  ) -> pd.DataFrame:
    """数据降采样

    使用简单的均匀采样策略

    Args:
        data: 原始数据
        target_points: 目标点数

    Returns:
        采样后的数据
    """
    if len(data) <= target_points:
      return data

    step = len(data) // target_points
    return data.iloc[::step].copy()

  def _save_plotly_figure(self, fig: go.Figure, output_path: Path) -> None:
    """保存 Plotly 图表

    Args:
        fig: Plotly Figure 对象
        output_path: 输出路径
    """
    try:
      output_path = Path(output_path)
      output_path.parent.mkdir(parents=True, exist_ok=True)

      if output_path.suffix == ".html":
        fig.write_html(str(output_path))
      elif output_path.suffix in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        fig.write_image(str(output_path))
      else:
        # 默认保存为 HTML
        fig.write_html(str(output_path.with_suffix(".html")))

      logger.info(f"Chart saved to {output_path}")

    except Exception as e:
      logger.error(f"Error saving chart to {output_path}: {e}")

  def plot_sleep_timeline(
    self,
    data: pd.DataFrame,
    title: str = "睡眠时间线",
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """绘制睡眠时间线图

    Args:
        data: 睡眠数据 (columns: start_date, end_date, value/stage)
        title: 图表标题
        output_path: 输出路径

    Returns:
        Plotly Figure 对象或 None
    """
    if data.empty:
      logger.warning("No data for sleep timeline")
      return None

    logger.info(f"Generating sleep timeline chart with {len(data)} sessions")

    try:
      # 睡眠阶段颜色映射
      stage_colors = {
        "Asleep": HEALTH_COLORS["primary"],
        "InBed": HEALTH_COLORS["light"],
        "Awake": HEALTH_COLORS["warning"],
        "Core": HEALTH_COLORS["info"],
        "Deep": HEALTH_COLORS["dark"],
        "REM": HEALTH_COLORS["secondary"],
      }

      fig = go.Figure()

      # 为每个睡眠阶段绘制条形
      for idx, row in data.iterrows():
        stage = row.get("value", "Asleep")
        if isinstance(stage, (int, float)):
          stage = "Asleep"

        fig.add_trace(
          go.Scatter(
            x=[row["start_date"], row["end_date"]],
            y=[idx, idx],
            mode="lines",
            line=dict(
              color=stage_colors.get(stage, HEALTH_COLORS["neutral"]),
              width=20,
            ),
            name=stage,
            showlegend=False,
            hovertemplate=f"<b>{stage}</b><br>"
            + f"开始: {row['start_date']}<br>"
            + f"结束: {row['end_date']}<br>"
            + "<extra></extra>",
          )
        )

      fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="睡眠会话",
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(f"Error generating sleep timeline: {e}")
      return None

  def plot_sleep_quality_trend(
    self,
    data: pd.DataFrame,
    title: str = "睡眠质量趋势",
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """绘制睡眠质量趋势图

    Args:
        data: 睡眠质量数据 (columns: date, duration, efficiency)
        title: 图表标题
        output_path: 输出路径

    Returns:
        Plotly Figure 对象或 None
    """
    if data.empty:
      logger.warning("No data for sleep quality trend")
      return None

    logger.info(f"Generating sleep quality trend chart with {len(data)} days")

    try:
      # 创建双Y轴图表
      fig = make_subplots(specs=[[{"secondary_y": True}]])

      # 睡眠时长
      fig.add_trace(
        go.Scatter(
          x=data["date"],
          y=data["total_duration"] / 60,  # 转换为小时
          mode="lines+markers",
          name="睡眠时长",
          line=dict(color=HEALTH_COLORS["primary"], width=2),
          marker=dict(size=6),
        ),
        secondary_y=False,
      )

      # 睡眠效率
      if "efficiency" in data.columns:
        fig.add_trace(
          go.Scatter(
            x=data["date"],
            y=data["efficiency"] * 100,  # 转换为百分比
            mode="lines+markers",
            name="睡眠效率",
            line=dict(color=HEALTH_COLORS["secondary"], width=2),
            marker=dict(size=6),
          ),
          secondary_y=True,
        )

      # 添加推荐睡眠时长线
      fig.add_hline(
        y=7,
        line_dash="dash",
        line_color=HEALTH_COLORS["success"],
        annotation_text="推荐时长: 7-9小时",
        annotation_position="right",
        secondary_y=False,
      )

      # 更新布局
      fig.update_xaxes(title_text="日期")
      fig.update_yaxes(title_text="睡眠时长 (小时)", secondary_y=False)
      fig.update_yaxes(title_text="睡眠效率 (%)", secondary_y=True)

      fig.update_layout(
        title=title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(f"Error generating sleep quality trend: {e}")
      return None

  def plot_sleep_stages_distribution(
    self,
    data: pd.DataFrame,
    title: str = "睡眠阶段分布",
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """绘制睡眠阶段分布饼图

    Args:
        data: 睡眠阶段数据 (columns: stage, duration)
        title: 图表标题
        output_path: 输出路径

    Returns:
        Plotly Figure 对象或 None
    """
    if data.empty:
      logger.warning("No data for sleep stages distribution")
      return None

    logger.info("Generating sleep stages distribution chart")

    try:
      # 聚合各阶段时长
      stage_durations = data.groupby("stage")["duration"].sum()

      # 睡眠阶段颜色
      colors = {
        "Asleep": HEALTH_COLORS["primary"],
        "Core": HEALTH_COLORS["info"],
        "Deep": HEALTH_COLORS["dark"],
        "REM": HEALTH_COLORS["secondary"],
        "Awake": HEALTH_COLORS["warning"],
        "InBed": HEALTH_COLORS["light"],
      }

      fig = go.Figure(
        data=[
          go.Pie(
            labels=stage_durations.index,
            values=stage_durations.values,
            marker=dict(
              colors=[
                colors.get(stage, HEALTH_COLORS["neutral"])
                for stage in stage_durations.index
              ]
            ),
            hovertemplate="<b>%{label}</b><br>"
            + "时长: %{value:.1f} 小时<br>"
            + "占比: %{percent}<br>"
            + "<extra></extra>",
          )
        ]
      )

      fig.update_layout(
        title=title,
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(f"Error generating sleep stages distribution: {e}")
      return None

  def plot_sleep_consistency(
    self,
    data: pd.DataFrame,
    title: str = "睡眠规律性分析",
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """绘制睡眠规律性分析图

    Args:
        data: 睡眠数据 (columns: date, bedtime, wake_time)
        title: 图表标题
        output_path: 输出路径

    Returns:
        Plotly Figure 对象或 None
    """
    if data.empty:
      logger.warning("No data for sleep consistency")
      return None

    logger.info(f"Generating sleep consistency chart with {len(data)} days")

    try:
      fig = go.Figure()

      # 入睡时间
      if "bedtime" in data.columns:
        # 将时间转换为小时数 (以24小时制表示)
        bedtime_hours = data["bedtime"].apply(
          lambda x: x.hour + x.minute / 60 if pd.notna(x) else None
        )

        fig.add_trace(
          go.Scatter(
            x=data["date"],
            y=bedtime_hours,
            mode="markers+lines",
            name="入睡时间",
            marker=dict(size=6, color=HEALTH_COLORS["primary"]),
            line=dict(color=HEALTH_COLORS["primary"], width=1.5),
          )
        )

      # 起床时间
      if "wake_time" in data.columns:
        wake_hours = data["wake_time"].apply(
          lambda x: x.hour + x.minute / 60 if pd.notna(x) else None
        )

        fig.add_trace(
          go.Scatter(
            x=data["date"],
            y=wake_hours,
            mode="markers+lines",
            name="起床时间",
            marker=dict(size=6, color=HEALTH_COLORS["secondary"]),
            line=dict(color=HEALTH_COLORS["secondary"], width=1.5),
          )
        )

      fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="时间 (24小时制)",
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(f"Error generating sleep consistency: {e}")
      return None

  def plot_weekday_vs_weekend_sleep(
    self,
    data: pd.DataFrame,
    title: str = "工作日 vs 周末睡眠对比",
    output_path: Path | None = None,
  ) -> go.Figure | None:
    """绘制工作日和周末睡眠对比图

    Args:
        data: 睡眠数据 (columns: date, duration, is_weekend)
        title: 图表标题
        output_path: 输出路径

    Returns:
        Plotly Figure 对象或 None
    """
    if data.empty:
      logger.warning("No data for weekday vs weekend sleep")
      return None

    logger.info("Generating weekday vs weekend sleep chart")

    try:
      # 分组数据
      weekday_data = data[~data["is_weekend"]]["duration"]
      weekend_data = data[data["is_weekend"]]["duration"]

      fig = go.Figure()

      # 工作日箱线图
      fig.add_trace(
        go.Box(
          y=weekday_data,
          name="工作日",
          marker_color=HEALTH_COLORS["primary"],
          boxmean="sd",
        )
      )

      # 周末箱线图
      fig.add_trace(
        go.Box(
          y=weekend_data,
          name="周末",
          marker_color=HEALTH_COLORS["secondary"],
          boxmean="sd",
        )
      )

      fig.update_layout(
        title=title,
        yaxis_title="睡眠时长 (小时)",
        width=self.width,
        height=self.height,
        template=PLOTLY_TEMPLATE,
      )

      if output_path:
        self._save_plotly_figure(fig, output_path)

      return fig

    except Exception as e:
      logger.error(f"Error generating weekday vs weekend sleep: {e}")
      return None

  def generate_heart_rate_report_charts(
    self,
    report: HeartRateAnalysisReport,
    output_dir: Path,
  ) -> dict[str, Path]:
    """生成心率报告的所有图表

    Args:
        report: 心率分析报告
        output_dir: 输出目录

    Returns:
        图表文件路径字典
    """
    logger.info("Generating heart rate report charts")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts: dict[str, Path] = {}

    # 这里暂时返回空字典，具体实现将在后续补充
    # 需要根据 HeartRateAnalysisReport 的具体数据结构来实现

    return charts

  def generate_sleep_report_charts(
    self,
    report: SleepAnalysisReport,
    output_dir: Path,
  ) -> dict[str, Path]:
    """生成睡眠报告的所有图表

    Args:
        report: 睡眠分析报告
        output_dir: 输出目录

    Returns:
        图表文件路径字典
    """
    logger.info("Generating sleep report charts")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts: dict[str, Path] = {}

    # 这里暂时返回空字典，具体实现将在后续补充
    # 需要根据 SleepAnalysisReport 的具体数据结构来实现

    return charts
