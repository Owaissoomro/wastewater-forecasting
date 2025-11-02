from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TableFormatConfig:
    """Configuration for Markdown table rendering."""
    max_rows: int = 50
    max_cols: int = 12
    float_decimals: int = 4
    index: bool = False
    na_rep: str = "NA"
    thousand_sep: Optional[str] = None  # e.g., "," to group thousands


def _escape_md(text: str) -> str:
    """Escape characters that can break Markdown tables."""
    return (
        text.replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "\\`")
    )


def _flatten_columns(columns: pd.Index) -> List[str]:
    """Flatten potential MultiIndex columns into a single-level list."""
    if isinstance(columns, pd.MultiIndex):
        return [" / ".join(map(str, tup)) for tup in columns.to_list()]
    return [str(c) for c in columns.to_list()]


def _format_number(
    x: float,
    decimals: int = 4,
    thousand_sep: Optional[str] = None,
) -> str:
    """Format a scalar number with robust rules for readability."""
    if not np.isfinite(x):
        return "NA"
    ax = abs(x)
    if (ax != 0 and ax < 10 ** (-(decimals + 1))) or ax >= 1e6:
        s = f"{x:.{decimals}e}"
    else:
        s = f"{x:.{decimals}f}"
    if thousand_sep is not None and "e" not in s and "." in s:
        # Insert thousand separators for integer part only
        int_part, frac_part = s.split(".")
        sign = ""
        if int_part.startswith("-"):
            sign, int_part = "-", int_part[1:]
        int_part = f"{int(int_part):,}".replace(",", thousand_sep)
        s = f"{sign}{int_part}.{frac_part}"
    return s


def _format_cell(
    v: Any,
    decimals: int,
    na_rep: str,
    thousand_sep: Optional[str],
) -> str:
    """Format a DataFrame cell into a Markdown-safe string."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na_rep
    if isinstance(v, (np.floating, float, np.integer, int)):
        return _escape_md(_format_number(float(v), decimals, thousand_sep))
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        try:
            return _escape_md(pd.to_datetime(v).date().isoformat())
        except Exception:
            return _escape_md(str(v))
    if isinstance(v, (list, tuple, set)):
        return _escape_md(", ".join(map(str, v)))
    return _escape_md(str(v))


def _truncate_df(
    df: pd.DataFrame, max_rows: int, max_cols: int
) -> Tuple[pd.DataFrame, bool, bool]:
    """Truncate a DataFrame deterministically to max_rows and max_cols with flags."""
    truncated_rows = False
    truncated_cols = False
    work = df.copy()

    # Stable column subset: first max_cols columns
    if work.shape[1] > max_cols:
        work = work.iloc[:, :max_cols]
        truncated_cols = True

    # Stable row subset: head(max_rows)
    if work.shape[0] > max_rows:
        work = work.head(max_rows)
        truncated_rows = True

    return work, truncated_rows, truncated_cols


def dataframe_to_markdown(
    df: pd.DataFrame,
    cfg: Optional[TableFormatConfig] = None,
) -> str:
    """Render a pandas DataFrame into a GitHub-flavored Markdown table.

    Parameters
    ----------
    df
        DataFrame to render.
    cfg
        Formatting configuration. Defaults are chosen for legibility.

    Returns
    -------
    str
        Markdown table as a string (including header and alignment row).
    """
    if cfg is None:
        cfg = TableFormatConfig()

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "_No KPIs available._"

    # Ensure deterministic column order and flatten MultiIndex
    df = df.copy()
    df.columns = _flatten_columns(df.columns)

    # Truncate safely
    df_trunc, trunc_r, trunc_c = _truncate_df(df, cfg.max_rows, cfg.max_cols)

    # Prepare header
    cols = list(df_trunc.columns)
    if cfg.index:
        header = ["index"] + cols
        body_vals = df_trunc.reset_index()
    else:
        header = cols
        body_vals = df_trunc

    # Build header row and alignment
    header_cells = [_escape_md(str(c)) for c in header]
    header_row = "| " + " | ".join(header_cells) + " |"
    align_row = "| " + " | ".join([":--" for _ in header]) + " |"

    # Build body rows
    rows: List[str] = []
    for _, row in body_vals.iterrows():
        cells = [
            _format_cell(v, cfg.float_decimals, cfg.na_rep, cfg.thousand_sep)
            for v in row.to_list()
        ]
        rows.append("| " + " | ".join(cells) + " |")

    table = "\n".join([header_row, align_row] + rows)
    notes = []
    if trunc_r:
        notes.append(f"- Only first {cfg.max_rows} rows shown.")
    if trunc_c:
        notes.append(f"- Only first {cfg.max_cols} columns shown.")
    notes.append(f"- Numeric values rounded to {cfg.float_decimals} decimals.")

    if notes:
        table = table + "\n\n" + "\n".join(notes)

    return table


def _format_insights(insights: Union[Sequence[str], Mapping[str, Any], None]) -> str:
    """Format insights into Markdown bullets or a key-value list."""
    if insights is None:
        return "_No insights recorded._"

    if isinstance(insights, Mapping):
        lines = []
        for k, v in insights.items():
            key = _escape_md(str(k))
            if isinstance(v, (list, tuple)):
                lines.append(f"- {key}:")
                for item in v:
                    lines.append(f"  - {_escape_md(str(item))}")
            else:
                lines.append(f"- {key}: {_escape_md(str(v))}")
        return "\n".join(lines)

    # Assume a sequence of strings
    if isinstance(insights, (list, tuple)):
        if not insights:
            return "_No insights recorded._"
        return "\n".join(f"- {_escape_md(str(x))}" for x in insights)

    # Fallback
    return f"- {_escape_md(str(insights))}"


def _format_links(links: Optional[Mapping[str, Union[str, Path, Sequence[Union[str, Path]]]]]) -> str:
    """Render artifact links into a Markdown bullet list."""
    if not links:
        return "_Artifacts will be populated after persistence._"

    lines: List[str] = []
    for label, pathlike in links.items():
        lab = _escape_md(str(label))
        if isinstance(pathlike, (list, tuple)):
            for p in pathlike:
                p_str = Path(p).as_posix()
                lines.append(f"- {lab}: `{p_str}`")
        else:
            p_str = Path(pathlike).as_posix()
            lines.append(f"- {lab}: `{p_str}`")
    return "\n".join(lines)


def stage_report(
    stage: str,
    insights: Union[Sequence[str], Mapping[str, Any], None],
    kpis_df: Optional[pd.DataFrame],
    links: Optional[Mapping[str, Union[str, Path, Sequence[Union[str, Path]]]]],
) -> str:
    """Compose a concise Markdown report for a pipeline stage.

    Parameters
    ----------
    stage
        Stage name, e.g., "preprocessing", "priors", "likelihood".
    insights
        Key findings or notes. Either a sequence of bullet points or a mapping
        from topic to message(s).
    kpis_df
        DataFrame of key performance indicators. Rendered as a Markdown table.
        MultiIndex columns are flattened. If None or empty, a placeholder is used.
    links
        Mapping from artifact labels to relative paths (or sequences of paths).
        Rendered as a bullet list for quick navigation.

    Returns
    -------
    str
        Markdown content suitable for persistence via ctx.write_report(...).

    Notes
    -----
    - The table rendering is deterministic and truncates large tables.
    - Paths are shown in POSIX style for cross-platform readability.
    """
    stage_clean = _escape_md(stage.strip())
    insights_md = _format_insights(insights)
    kpi_table_md = dataframe_to_markdown(kpis_df)
    links_md = _format_links(links)

    sections = [
        f"# Stage report: {stage_clean}",
        "",
        "## Insights",
        insights_md,
        "",
        "## KPI summary",
        kpi_table_md,
        "",
        "## Artifacts",
        links_md,
        "",
        "_This report was generated programmatically; figures and tables are saved "
        f"under results/{stage_clean}/.*_",
        "",
    ]
    return "\n".join(sections)

# --- Simple "sections → Markdown" helper (used by some stages) -----------------
def render_markdown_sections(
    sections: Union[
        Mapping[str, Union[str, pd.DataFrame]],
        Sequence[Tuple[str, Union[str, pd.DataFrame]]]
    ],
    heading_level: int = 2,
    table_cfg: Optional[TableFormatConfig] = None,
) -> str:
    """
    Render a mapping or sequence of (title, content) into Markdown.
    Content may be a plain string or a pandas DataFrame (rendered as a table).

    Parameters
    ----------
    sections : mapping or sequence
        Either {title: content} or [(title, content), ...].
    heading_level : int, default 2
        Base heading level (2 → '## ').
    table_cfg : TableFormatConfig, optional
        Formatting options for DataFrame → Markdown rendering.

    Returns
    -------
    str
        Markdown document string.
    """
    # Normalize to ordered list of pairs
    if isinstance(sections, Mapping):
        items: List[Tuple[str, Union[str, pd.DataFrame]]] = list(sections.items())
    else:
        items = list(sections)

    lines: List[str] = []
    lvl = max(1, min(int(heading_level), 6))
    for title, content in items:
        lines.append(f"{'#' * lvl} {_escape_md(str(title))}")
        if isinstance(content, pd.DataFrame):
            lines.append(dataframe_to_markdown(content, cfg=table_cfg))
        elif content is None:
            lines.append("_No content._")
        else:
            lines.append(_escape_md(str(content)))
        lines.append("")  # blank line after each section

    return "\n".join(lines).rstrip() + "\n"
