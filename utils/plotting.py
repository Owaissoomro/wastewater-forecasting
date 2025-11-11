# -*- coding: utf-8 -*-
"""
utils.plotting 
==========================================================

-------
set_matplotlib_style
place_legend_below
continuous_cmap
categorical_palette
save_figure

Plus useful extras (kept lightweight):
format_date_axis, mosaic_figure, gridspec_figure, finalize_figure, plot_with_ci,
heatmap, kde_curve, get_color_cycle, add_zero_line, annotate_panel_label,
rugplot, bounded_fill_between
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union, List

import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.colors import Colormap, to_rgba
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

# ---------- Optional dependencies (graceful fallbacks) ----------
try:
    # SciPy only if available (for KDE)
    from scipy.stats import gaussian_kde  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    import seaborn as sns  # style/palettes
    _HAVE_SEABORN = True
except Exception:
    _HAVE_SEABORN = False

# ---------- Public API ----------

__all__ = [
    "set_matplotlib_style",
    "place_legend_below",
    "continuous_cmap",
    "categorical_palette",
    "save_figure",
    # extras
    "format_date_axis",
    "mosaic_figure",
    "gridspec_figure",
    "finalize_figure",
    "plot_with_ci",
    "heatmap",
    "kde_curve",
    "get_color_cycle",
    "add_zero_line",
    "annotate_panel_label",
    "rugplot",
    "bounded_fill_between",
]


# ---------- Core styling helpers ----------

def set_matplotlib_style(cfg: Optional[Mapping[str, Any]] = None) -> None:
    """
    Apply a modern, publication-friendly Matplotlib style with safe fallbacks.

    Parameters
    ----------
    cfg : dict, optional
        Optional rcParams overrides. Accepts keys like 'font.size', 'figure.dpi', etc.
        If present, values in cfg take precedence over defaults below.
    """
    # Prefer SciencePlots if installed (no LaTeX required)
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex", "grid"])
    except Exception:
        plt.style.use("default")

    # Layer seaborn theme if available
    if _HAVE_SEABORN:
        try:
            sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
        except Exception:
            pass

    # Baseline rc defaults (neutral look, subtle grid)
    defaults: Dict[str, Any] = {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "font.size": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.color": "0.85",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fancybox": True,
        "legend.borderaxespad": 0.6,
        "lines.linewidth": 2.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
    if cfg:
        # Allow both top-level and cfg["plotting"] overrides
        if "plotting" in cfg and isinstance(cfg["plotting"], Mapping):
            defaults.update({k: v for k, v in cfg["plotting"].items() if v is not None})
        for k, v in cfg.items():
            if k in mpl.rcParams:
                defaults[k] = v

    # Optional color cycle override
    color_cycle = defaults.pop("color.cycle", None)
    mpl.rcParams.update(defaults)  # type: ignore[arg-type]
    if isinstance(color_cycle, (list, tuple)) and len(color_cycle) > 0:
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color_cycle)


def place_legend_below(
    ax: Axes,
    ncol: int = 3,
    pad: float = 0.02,
) -> Optional[Legend]:
    """
    Place a compact legend centered *below* the axes.

    Parameters
    ----------
    ax : Axes
        Target axes.
    ncol : int, default 3
        Columns in the legend (auto-capped to number of labels).
    pad : float, default 0.02
        Vertical padding (in axes fraction).

    Returns
    -------
    Legend or None
    """
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return None
    cols = max(1, min(ncol, len(labels)))
    return ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -pad),
        ncol=cols,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.3
    )


def continuous_cmap(name: str = "cet_fire") -> Colormap:
    """
    Get a continuous colormap, preferring ColorCET if installed.

    Falls back to Matplotlib's registry and then 'viridis'.
    """
    try:
        import colorcet as cc  # type: ignore
        if hasattr(cc, "mpl_colormap"):
            return cc.mpl_colormap(name)  # e.g., 'cet_fire', 'cet_bmw'
        # Some versions expose colormaps via plt.cm registry
        return plt.cm.get_cmap(name)
    except Exception:
        try:
            return plt.cm.get_cmap(name)
        except Exception:
            return plt.cm.get_cmap("viridis")


def categorical_palette(n: int) -> List[Tuple[float, float, float, float]]:
    """
    Distinct categorical palette with graceful fallbacks.

    Order of preference:
    1) colorcet.glasbey (many distinct colors)
    2) seaborn colorblind palette
    3) Matplotlib tab20 sampled cyclically
    """
    if n <= 0:
        return []
    try:
        import colorcet as cc  # type: ignore
        base = list(cc.glasbey)
        if n <= len(base):
            return [to_rgba(c) for c in base[:n]]
        return [to_rgba(base[i % len(base)]) for i in range(n)]
    except Exception:
        if _HAVE_SEABORN:
            try:
                return [to_rgba(c) for c in sns.color_palette("colorblind", n_colors=max(3, n))]
            except Exception:
                pass
        cm = plt.cm.get_cmap("tab20")
        return [cm(i) for i in np.linspace(0.05, 0.95, n)]


def save_figure(
    fig: Figure,
    path: str,
    tight: bool = True,
    bbox_inches: str = "tight",
    pad_inches: float = 0.04,
) -> None:
    """
    Save a figure with consistent settings and a white background.
    """
    try:
        if tight:
            try:
                fig.tight_layout()
            except Exception:
                pass
        fig.savefig(
            path,
            dpi=mpl.rcParams.get("savefig.dpi", 140),
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            facecolor="white"
        )
    except Exception as e:
        warnings.warn(f"Failed to save figure {path}: {e}")


# ---------- Useful extras (kept API-stable) ----------

def format_date_axis(ax: Axes, rotation: int = 0, ha: str = "center",
                     interval: Optional[int] = None, tz: Optional[str] = None) -> None:
    """Format x-axis as YYYY-MM-DD with reasonable tick count."""
    locator = (mdates.DayLocator(interval=interval, tz=tz) if interval and interval > 0
               else mdates.AutoDateLocator(minticks=4, maxticks=10, tz=tz))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz=tz))
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment(ha)


def mosaic_figure(layout: Union[str, List[List[str]], List[str]],
                  figsize: Tuple[float, float] = (8.0, 5.0),
                  sharex: bool = False, sharey: bool = False,
                  dpi: Optional[int] = None) -> Tuple[Figure, Dict[str, Axes]]:
    """Create a figure via subplot_mosaic; returns (fig, axes_dict)."""
    fig = plt.figure(figsize=figsize, dpi=dpi or mpl.rcParams.get("figure.dpi", 140))
    axes = fig.subplot_mosaic(layout, sharex=sharex, sharey=sharey)
    return fig, axes


def gridspec_figure(nrows: int, ncols: int, figsize: Tuple[float, float] = (8.0, 5.0),
                    height_ratios: Optional[Sequence[float]] = None,
                    width_ratios: Optional[Sequence[float]] = None,
                    hspace: float = 0.25, wspace: float = 0.25,
                    dpi: Optional[int] = None) -> Tuple[Figure, List[Axes]]:
    """Create a figure via GridSpec; returns (fig, axes_list)."""
    fig = plt.figure(figsize=figsize, dpi=dpi or mpl.rcParams.get("figure.dpi", 140))
    gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig,
                  height_ratios=height_ratios, width_ratios=width_ratios,
                  hspace=hspace, wspace=wspace)
    axes: List[Axes] = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]
    return fig, axes


def finalize_figure(ctx: Any, fig: Figure, name: str,
                    width: Optional[float] = None, height: Optional[float] = None,
                    metadata: Optional[Mapping[str, Any]] = None, close: bool = True) -> None:
    """Finalize & persist a figure via ctx.write_figure(fig, name, metadata=...)."""
    if width or height:
        cw, ch = fig.get_size_inches()
        fig.set_size_inches(width or cw, height or ch, forward=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    if metadata is not None:
        ctx.write_figure(name, fig, metadata=metadata)  # allow ctx(name, fig) or ctx(fig, name)
    else:
        # Try both call signatures to be tolerant
        try:
            ctx.write_figure(name, fig)
        except Exception:
            ctx.write_figure(fig, name)
    if close:
        plt.close(fig)


def plot_with_ci(ax: Axes, x: Sequence[float], y: Sequence[float],
                 y_lower: Optional[Sequence[float]] = None,
                 y_upper: Optional[Sequence[float]] = None,
                 color: Optional[Union[str, Tuple[float, float, float, float]]] = None,
                 label: Optional[str] = None,
                 alpha_fill: float = 0.2,
                 linewidth: Optional[float] = None) -> None:
    """Plot a line with optional confidence band."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    lw = linewidth or float(mpl.rcParams.get("lines.linewidth", 2.0))
    ln, = ax.plot(x_arr, y_arr, color=color, label=label, linewidth=lw)
    fill_color = color or ln.get_color()
    if y_lower is not None and y_upper is not None:
        lo = np.minimum(np.asarray(y_lower), np.asarray(y_upper))
        hi = np.maximum(np.asarray(y_lower), np.asarray(y_upper))
        ax.fill_between(x_arr, lo, hi, color=fill_color, alpha=alpha_fill, linewidth=0.0)


def heatmap(ax: Axes, data: np.ndarray,
            xlabels: Optional[Sequence[str]] = None,
            ylabels: Optional[Sequence[str]] = None,
            cmap: Union[str, Colormap] = "viridis",
            cbar: bool = True, cbar_label: Optional[str] = None,
            vmin: Optional[float] = None, vmax: Optional[float] = None,
            show_values: bool = False, fmt: str = ".2f", text_color: str = "k") -> None:
    """Draw a heatmap with optional labels and colorbar."""
    data = np.asarray(data)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    if ylabels is not None:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.grid(False)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cb.set_label(cbar_label)
    if show_values:
        fmt_fn = ("{:" + fmt + "}").format
        ny, nx = data.shape
        for i in range(ny):
            for j in range(nx):
                ax.text(j, i, fmt_fn(float(data[i, j])), ha="center", va="center", color=text_color)


def kde_curve(values: Sequence[float], grid: Optional[np.ndarray] = None,
              bw_factor: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """KDE curve for 1D data (SciPy) or a normalized histogram fallback."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    if grid is None:
        lo, hi = np.quantile(arr, [0.001, 0.999])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = (arr.min(), arr.max() + np.finfo(float).eps)
        grid = np.linspace(lo, hi, 512)
    if _HAVE_SCIPY:
        kde = gaussian_kde(arr)
        if bw_factor is not None and bw_factor > 0:
            try:
                kde.set_bandwidth(kde.factor * bw_factor)
            except Exception:
                pass
        y = kde(grid)
        y = y / np.trapz(y, grid) if np.trapz(y, grid) > 0 else y
        return grid, y
    # histogram fallback
    counts, edges = np.histogram(arr, bins=50, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def get_color_cycle(n: int, cmap: Union[str, Colormap] = "tab10") -> List[Tuple[float, float, float, float]]:
    """Return n RGBA colors from current cycle or a fallback colormap."""
    if n <= 0:
        return []
    cycle = mpl.rcParams.get("axes.prop_cycle", None)
    if cycle is not None:
        try:
            cols = [to_rgba(c["color"]) for c in cycle]
            if len(cols) >= n:
                return cols[:n]
        except Exception:
            pass
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return [cm(i) for i in np.linspace(0.05, 0.95, n)]


def add_zero_line(ax: Axes, axis: str = "y", color: str = "0.6",
                  linewidth: float = 1.0, linestyle: str = "--") -> None:
    """Add a zero reference line along x or y axis."""
    if axis == "y":
        ax.axhline(0.0, color=color, linewidth=linewidth, linestyle=linestyle, zorder=0)
    elif axis == "x":
        ax.axvline(0.0, color=color, linewidth=linewidth, linestyle=linestyle, zorder=0)


def annotate_panel_label(ax: Axes, label: str, x: float = -0.02, y: float = 1.02,
                         weight: str = "bold") -> None:
    """Annotate a subplot with a bold panel label (e.g., 'A', 'B')."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=mpl.rcParams.get("axes.titlesize", 12),
            fontweight=weight, va="bottom", ha="right")


def rugplot(ax: Axes, values: Sequence[float], axis: str = "x",
            height: float = 0.05, color: Optional[str] = None, alpha: float = 0.6) -> None:
    """Draw a rug plot along x or y axis."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    if axis == "x":
        ymin, ymax = ax.get_ylim()
        dy = (ymax - ymin) * height
        for v in vals:
            ax.plot([v, v], [ymin, ymin + dy], color=color, alpha=alpha, linewidth=1.0, solid_capstyle="butt")
    else:
        xmin, xmax = ax.get_xlim()
        dx = (xmax - xmin) * height
        for v in vals:
            ax.plot([xmin, xmin + dx], [v, v], color=color, alpha=alpha, linewidth=1.0, solid_capstyle="butt")


def bounded_fill_between(ax: Axes, x: Sequence[float], y1: Sequence[float], y2: Sequence[float],
                         lower: Optional[float] = None, upper: Optional[float] = None, **kwargs: Any) -> None:
    """Fill between two curves with optional clipping bounds."""
    x_arr = np.asarray(x)
    a = np.asarray(y1, dtype=float)
    b = np.asarray(y2, dtype=float)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    if lower is not None:
        lo = np.maximum(lo, lower); hi = np.maximum(hi, lower)
    if upper is not None:
        lo = np.minimum(lo, upper); hi = np.minimum(hi, upper)
    ax.fill_between(x_arr, lo, hi, **kwargs)
