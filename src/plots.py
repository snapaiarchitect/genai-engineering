"""
Standardized plotting engine for all sierra-genai-engineering projects.
Consistent styling, fonts, colors, and export paths.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger("genai_plots")

# ---- Style Configuration ----
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#0f172a",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": "#e2e8f0",
    "savefig.facecolor": "#0f172a",
    "axes.grid": True,
    "grid.color": "#1e293b",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

# Color palette matching the portfolio dark theme
PALETTE = [
    "#06b6d4",  # cyan
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#ef4444",  # red
    "#84cc16",  # lime
    "#6366f1",  # indigo
]

sns.set_palette(PALETTE)


def get_figures_dir(project_root: str) -> str:
    """Return the standard figures/ directory for a project."""
    fig_dir = os.path.join(project_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def save_figure(
    fig: plt.Figure,
    filename: str,
    project_root: str,
    formats: Tuple[str, ...] = ("png",),
    bbox_inches: str = "tight",
    pad_inches: float = 0.2,
) -> Dict[str, str]:
    """
    Save a matplotlib figure to the project's figures/ directory.
    Returns a dict mapping format to saved path.
    """
    fig_dir = get_figures_dir(project_root)
    saved = {}

    for fmt in formats:
        if not fmt.startswith("."):
            fmt = f".{fmt}"
        path = os.path.join(fig_dir, f"{filename}{fmt}")
        fig.savefig(path, bbox_inches=bbox_inches, pad_inches=pad_inches)
        saved[fmt] = path
        logger.info(f"[PLOT] Saved {path}")

    return saved


def plot_horizontal_bar(
    data: Dict[str, Any],
    title: str,
    xlabel: str = "Count",
    ylabel: str = "",
    color: str = PALETTE[0],
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Create a standardized horizontal bar chart."""
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(data.keys())
    values = list(data.values())

    bars = ax.barh(labels, values, color=color)
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}",
            va="center",
            fontsize=9,
            color="#e2e8f0",
        )

    plt.tight_layout()
    return fig


def plot_timeline(
    dates: Any,
    values: Any,
    title: str,
    xlabel: str = "Date",
    ylabel: str = "Count",
    color: str = PALETTE[0],
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Create a standardized timeline / time-series chart."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, values, color=color, linewidth=2, marker="o", markersize=4)
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.fill_between(dates, values, alpha=0.15, color=color)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_pie(
    data: Dict[str, Any],
    title: str,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Create a standardized pie chart."""
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(data.keys())
    values = list(data.values())
    colors = PALETTE[: len(labels)]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.75,
    )
    for autotext in autotexts:
        autotext.set_color("#0f172a")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)

    ax.set_title(title, fontweight="bold", pad=15)
    plt.tight_layout()
    return fig
