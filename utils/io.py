from __future__ import annotations

import io as _io
import json
import os
import tempfile
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional plotting helpers (harmless no-ops if unavailable)
try:
    from utils.plotting import set_matplotlib_style, place_legend_below  # type: ignore
except Exception:  # pragma: no cover
    def set_matplotlib_style() -> None:
        return
    def place_legend_below(ax: plt.Axes, ncol: Optional[int] = None) -> None:
        return


# ------------------------
# Path helpers
# ------------------------

def project_root() -> Path:
    """Return the repository root directory by searching upward for markers."""
    here = Path(__file__).resolve()
    candidates = list(here.parents)
    markers = {"configs", "utils", "stages", "results"}
    for p in candidates:
        if all((p / m).exists() for m in markers):
            return p
    # Fallback to repo root guessed from utils/
    return here.parents[1]


def rpath(*parts: Union[str, Path]) -> Path:
    """Return a path anchored at the project root."""
    root = project_root()
    if any(Path(p).is_absolute() for p in parts):
        raise ValueError("rpath() does not accept absolute paths.")
    full = root.joinpath(*map(str, parts)).resolve()
    try:
        full.relative_to(root)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Resolved path escapes project root: {full}") from e
    return full


# ------------------------
# Internal utilities
# ------------------------

def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _fsync_dir(dpath: Path) -> None:
    try:
        fd = os.open(str(dpath), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        # Best-effort, ignore on platforms without directory fsync
        pass


def _atomic_replace(src: Path, dst: Path) -> None:
    os.replace(str(src), str(dst))
    _fsync_dir(dst.parent)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomic write of bytes to file path."""
    _ensure_dir(path)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmp:
        tmp_name = Path(tmp.name)
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
    _atomic_replace(tmp_name, path)


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomic write of text to file path."""
    _ensure_dir(path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding=encoding, newline="\n") as tmp:
        tmp_name = Path(tmp.name)
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
    _atomic_replace(tmp_name, path)


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _to_jsonable(obj: Any) -> Any:
    """Convert objects to JSON-serializable equivalents."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # pandas / numpy
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return obj.isoformat()
    # Path
    if isinstance(obj, Path):
        return str(obj)
    # Fallback to string
    return str(obj)


def _infer_timeseries_sort_keys(columns: Iterable[str]) -> Tuple[str, ...]:
    cols = set(columns)
    keys = []
    for k in ("site_id", "date", "lineage"):
        if k in cols:
            keys.append(k)
    return tuple(keys)


def _coerce_datestring(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        s = pd.to_datetime(df["date"], errors="coerce", utc=False)
        df = df.copy()
        df["date"] = s.dt.strftime("%Y-%m-%d")
    return df


def _tidy_sort(df: pd.DataFrame) -> pd.DataFrame:
    keys = _infer_timeseries_sort_keys(df.columns)
    if len(keys) > 0:
        return df.sort_values(list(keys), kind="mergesort", ignore_index=True)
    return df


# ------------------------
# Public export API (CSV only)
# ------------------------

def export_metrics(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    *,
    float_format: str = "%.6g",
) -> Dict[str, Any]:
    """Write a metrics DataFrame to CSV atomically.

    Notes
    -----
    - Rows are sorted by (site_id, date, lineage) if present.
    - 'date' is coerced to YYYY-MM-DD if the column exists.
    """
    base_path = Path(base_path)
    _ensure_dir(base_path.with_suffix(".csv"))

    df = _coerce_datestring(_tidy_sort(df)).copy()

    # CSV
    csv_path = base_path.with_suffix(".csv")
    csv_buf = _io.StringIO()
    df.to_csv(csv_buf, index=False, encoding="utf-8", float_format=float_format)
    _atomic_write_text(csv_path, csv_buf.getvalue())

    artifacts: Dict[str, Any] = {
        "kind": "metrics",
        "created": _now_iso(),
        "csv": {
            "path": str(csv_path.relative_to(project_root())),
            "sha256": _file_sha256(csv_path),
            "bytes": csv_path.stat().st_size,
        },
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(map(str, df.columns)),
    }
    return artifacts


def export_table(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    *,
    float_format: str = "%.6g",
) -> Dict[str, Any]:
    """Write a table to CSV atomically."""
    base_path = Path(base_path)
    _ensure_dir(base_path.with_suffix(".csv"))

    artifacts = export_metrics(
        df=_coerce_datestring(_tidy_sort(df)),
        base_path=base_path,
        float_format=float_format,
    )
    artifacts["kind"] = "table"
    return artifacts


def export_report_md(text: str, path: Union[str, Path]) -> Dict[str, Any]:
    """Write a Markdown report atomically."""
    path = Path(path)
    if path.suffix.lower() != ".md":
        path = path.with_suffix(".md")
    _ensure_dir(path)
    _atomic_write_text(path, text)

    return {
        "kind": "report",
        "created": _now_iso(),
        "path": str(path.relative_to(project_root())),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def export_figure(
    fig: plt.Figure,
    base_path: Union[str, Path],
    *,
    dpi: int = 200,
    transparent: bool = False,
    apply_style: bool = True,
    place_legends: bool = True,
) -> Dict[str, Any]:
    """Save a Matplotlib figure to PDF and PNG atomically."""
    if apply_style:
        set_matplotlib_style()

    if place_legends:
        try:
            for ax in fig.get_axes():
                handles, labels = ax.get_legend_handles_labels()
                if any(lbl for lbl in labels):
                    place_legend_below(ax)
        except Exception:
            pass

    try:
        fig.tight_layout()
    except Exception:
        pass

    base_path = Path(base_path)
    _ensure_dir(base_path.with_suffix(".pdf"))

    pdf_path = base_path.with_suffix(".pdf")
    png_path = base_path.with_suffix(".png")

    pdf_buffer = _io.BytesIO()
    png_buffer = _io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", transparent=transparent, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_buffer, format="png", dpi=dpi, transparent=transparent, bbox_inches="tight", pad_inches=0.02)

    _atomic_write_bytes(pdf_path, pdf_buffer.getvalue())
    _atomic_write_bytes(png_path, png_buffer.getvalue())

    artifacts = {
        "kind": "figure",
        "created": _now_iso(),
        "pdf": {
            "path": str(pdf_path.relative_to(project_root())),
            "sha256": _file_sha256(pdf_path),
            "bytes": pdf_path.stat().st_size,
        },
        "png": {
            "path": str(png_path.relative_to(project_root())),
            "sha256": _file_sha256(png_path),
            "bytes": png_path.stat().st_size,
        },
        "size_inches": {
            "width": float(fig.get_size_inches()[0]),
            "height": float(fig.get_size_inches()[1]),
        },
        "dpi": int(dpi),
    }
    return artifacts


def append_jsonl(path: Union[str, Path], record: Mapping[str, Any]) -> Dict[str, Any]:
    """Append a JSON object as a single line to a .jsonl log atomically."""
    path = Path(path)
    if path.suffix.lower() != ".jsonl":
        path = path.with_suffix(".jsonl")
    _ensure_dir(path)

    base: Dict[str, Any] = {
        "time": _now_iso(),
        "level": "INFO",
        "stage": None,
        "site_id": None,
        "lineage": None,
        "message": None,
        "context": None,
    }
    base.update({str(k): v for k, v in record.items()})
    jsonable = _to_jsonable(base)
    line = json.dumps(jsonable, ensure_ascii=False) + "\n"
    data = line.encode("utf-8")

    with open(path, "ab") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    return {
        "kind": "log_append",
        "path": str(path.relative_to(project_root())),
        "bytes_appended": len(data),
        "time": _now_iso(),
    }


# ------------------------
# Higher-level helpers for RunContext
# ------------------------

def prepare_stage_dirs(stage: str) -> Dict[str, Path]:
    """Prepare canonical result directories for a stage."""
    root = rpath("results", stage)
    dirs = {
        "root": root,
        "metrics": root / "metrics",
        "tables": root / "tables",
        "figures": root / "figures",
        "logs": root / "logs",
    }
    for p in dirs.values():
        if p.name != "root":
            p.mkdir(parents=True, exist_ok=True)
    return dirs


def artifact_entry(path: Path, kind: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a manifest entry for a file on disk."""
    entry = {
        "kind": kind,
        "path": str(path.relative_to(project_root())),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
        "created": _now_iso(),
    }
    if extra:
        entry["extra"] = {k: _to_jsonable(v) for k, v in extra.items()}
    return entry


def export_detector_timeseries(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    *,
    float_format: str = "%.6g",
) -> Dict[str, Any]:
    """Export detector raw timeseries with canonical schema (CSV only).

    Schema
    ------
    site_id : str
    date : YYYY-MM-DD
    lineage : str
    value : float
    threshold_crossed : bool
    """
    required = {"site_id", "date", "lineage", "value", "threshold_crossed"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Detector timeseries missing required columns: {sorted(missing)}")
    df2 = df.copy()
    df2["threshold_crossed"] = df2["threshold_crossed"].astype(bool)
    return export_metrics(
        df=_tidy_sort(_coerce_datestring(df2)),
        base_path=base_path,
        float_format=float_format,
    )


def export_residuals(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    *,
    float_format: str = "%.6g",
) -> Dict[str, Any]:
    """Export residuals with canonical schema (CSV only).

    Schema
    ------
    site_id : str
    date : YYYY-MM-DD
    mutation : str
    obs_af : float
    pred_af : float
    resid : float
    resid_std : float
    """
    required = {"site_id", "date", "mutation", "obs_af", "pred_af", "resid", "resid_std"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Residuals missing required columns: {sorted(missing)}")
    df2 = df.copy()
    for c in ["obs_af", "pred_af", "resid", "resid_std"]:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return export_metrics(
        df=_tidy_sort(_coerce_datestring(df2)),
        base_path=base_path,
        float_format=float_format,
    )
