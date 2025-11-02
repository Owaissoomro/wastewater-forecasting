

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import io
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Mapping
from pathlib import Path

import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow flat, non-package repo imports from anywhere following the project rule
# Scripts/notebooks should already sys.path.append(root), but ensure relative import works here too.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Optional utilities (config, plotting, provenance, seeds); guard to degrade gracefully if absent
with contextlib.suppress(Exception):
    from utils import config as ucfg  # type: ignore
with contextlib.suppress(Exception):
    from utils import plotting as uplot  # type: ignore
with contextlib.suppress(Exception):
    from utils import provenance as uprov  # type: ignore
with contextlib.suppress(Exception):
    from utils import logging as ulog  # type: ignore  # structured logging helpers (optional)
with contextlib.suppress(Exception):
    from utils import seeds as useeds  # type: ignore


def _now_utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sanitize_name(name: str) -> str:
    keep = [c if c.isalnum() or c in ("-", "_", ".", "+") else "_" for c in name]
    out = "".join(keep).strip("._")
    return out or "unnamed"


def _to_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, pd.Series):
        return obj.to_frame().reset_index()
    if isinstance(obj, Mapping):
        return pd.DataFrame([obj])
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return pd.DataFrame()
        if isinstance(obj[0], Mapping):
            return pd.DataFrame(obj)
        return pd.DataFrame({ "value": list(obj) })
    if np.isscalar(obj):
        return pd.DataFrame({ "value": [obj] })
    # Fallback try DataFrame constructor directly
    return pd.DataFrame(obj)


def _jsonify(x: Any) -> Any:
    # Convert numpy / pandas dtypes into JSON-friendly types
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, (pd.Timedelta,)):
        return str(x)
    if isinstance(x, (pd.Series,)):
        return _jsonify(x.to_dict())
    if isinstance(x, (pd.DataFrame,)):
        return _jsonify(x.to_dict(orient="records"))
    if isinstance(x, (dict,)):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    with contextlib.suppress(Exception):
        return json.loads(json.dumps(x))
    return str(x)


def _get_seed_from_config(cfg: Mapping[str, Any], default_seed: int = 12345) -> int:
    # Accept several common placements
    if "seed" in cfg and isinstance(cfg["seed"], (int, np.integer)):
        return int(cfg["seed"])
    for k in ("run", "runtime", "experiment"):
        if isinstance(cfg.get(k, None), Mapping):
            v = cfg[k].get("seed", None)
            if isinstance(v, (int, np.integer)):
                return int(v)
    return int(default_seed)


def _matplotlib_style_safe() -> None:
    # Ensure Matplotlib is in a consistent publication-friendly style
    try:
        if "uplot" in globals() and hasattr(uplot, "set_matplotlib_style"):
            uplot.set_matplotlib_style()
        else:
            # Minimal sensible defaults
            matplotlib.rcParams.update({
                "figure.dpi": 100,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "axes.grid": True,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 9,
                "figure.autolayout": False,
            })
    except Exception:
        pass


def _set_global_seeds_safe(seed: int) -> None:
    # Prefer utils.seeds if present
    try:
        if "useeds" in globals() and hasattr(useeds, "set_global_seeds"):
            useeds.set_global_seeds(seed)
            return
    except Exception:
        pass
    # Fallback minimal deterministic seeds
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

@dataclass
class FileRecord:
    path: str
    kind: str
    sha256: str
    bytes: int
    created_at: str
    extra: Dict[str, Any] = field(default_factory=dict)

# ====== BEGIN: REPLACEMENT StageContext (utils/run.py) ======
class StageContext:
    """
    Stage-specific context for writing metrics, tables, figures, reports, and logs.

    Created via RunContext.stage(name).
    Ensures Save-Everything policy and structured JSONL logging.
    """

    def __init__(self, run: "RunContext", name: str) -> None:
        self.run = run
        self.name = _sanitize_name(name)
        _matplotlib_style_safe()

        self.stage_dir = self.run.results_root / self.name
        self.run_stage_dir = self.run.runs_root / self.run.run_id / self.name

        # Subdirectories (both global stage and run-scoped)
        self.metrics_dir = self.stage_dir / "metrics"
        self.tables_dir = self.stage_dir / "tables"
        self.figures_dir = self.stage_dir / "figures"
        self.logs_dir = self.stage_dir / "logs"

        self.run_metrics_dir = self.run_stage_dir / "metrics"
        self.run_tables_dir = self.run_stage_dir / "tables"
        self.run_figures_dir = self.run_stage_dir / "figures"
        self.run_logs_dir = self.run_stage_dir / "logs"

        for d in (
            self.metrics_dir, self.tables_dir, self.figures_dir, self.logs_dir,
            self.run_metrics_dir, self.run_tables_dir, self.run_figures_dir, self.run_logs_dir,
        ):
            _ensure_dir(d)

        # Report file paths
        self.report_path = self.stage_dir / "report.md"
        self.run_report_path = self.run_stage_dir / "report.md"

        self._manifest: List[FileRecord] = []
        self._start_time = _now_utc_iso()
        self._closed = False

        # Logging handles
        self._log_fp: Optional[io.TextIOBase] = None
        self._run_log_fp: Optional[io.TextIOBase] = None
        self._open_logs()

        # Per-stage seed to ensure determinism (derive from run seed and stage name)
        stage_seed = (self.run.seed * 1315423911) ^ (hash(self.name) & 0xFFFFFFFF)
        _set_global_seeds_safe(stage_seed)

        # Log stage start
        self.log(level="INFO", message="Stage started",
                 context={"run_id": self.run.run_id,
                          "stage_seed": stage_seed,
                          "cfg_digest": self.run.cfg_digest})

    # --------------------------- Logging ---------------------------

    def _open_logs(self) -> None:
        """Open JSONL logs under both locations."""
        self._log_fp = (self.logs_dir / f"{self.name}.jsonl").open("a", encoding="utf-8")
        self._run_log_fp = (self.run_logs_dir / f"{self.name}.jsonl").open("a", encoding="utf-8")

    def _close_logs(self) -> None:
        with contextlib.suppress(Exception):
            if self._log_fp:
                self._log_fp.flush(); self._log_fp.close()
        with contextlib.suppress(Exception):
            if self._run_log_fp:
                self._run_log_fp.flush(); self._run_log_fp.close()
        self._log_fp = None
        self._run_log_fp = None

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        js = json.dumps(record, ensure_ascii=False)
        if self._log_fp is not None:
            self._log_fp.write(js + "\n"); self._log_fp.flush()
        if self._run_log_fp is not None:
            self._run_log_fp.write(js + "\n"); self._run_log_fp.flush()

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure required fields and fold stray keys into 'context'."""
        r = dict(record)
        # Handle historical bug: payload mistakenly passed as level=dict(...)
        if isinstance(r.get("level"), dict) and not r.get("message"):
            inner = r["level"]
            if isinstance(inner, dict):
                r = dict(inner)

        r.setdefault("time", _now_utc_iso())
        r.setdefault("level", "INFO")
        r.setdefault("stage", self.name)
        r.setdefault("site_id", None)
        r.setdefault("lineage", None)
        r.setdefault("message", "")

        ctx = r.get("context")
        if ctx is None:
            r["context"] = {}
        elif not isinstance(ctx, dict):
            r["context"] = {"value": ctx}

        allowed = {"time", "level", "stage", "site_id", "lineage", "message", "context"}
        extras = {k: v for k, v in r.items() if k not in allowed}
        if extras:
            r["context"].update(extras)
            for k in list(extras.keys()):
                r.pop(k, None)
        return r

    def log(self, *args, **kwargs) -> None:
        """
        Accepts:
          - log("message", level="INFO", site_id=..., lineage=..., context={...})
          - log(level="INFO", message="...", context={...})
          - log({...})  # fully-formed payload dict
          - log(level={...})  # historical bug: entire payload passed as 'level'
        Any extra kwargs are merged into 'context'. Never raises.
        """
        try:
            # Case: dict payload passed as the first arg
            if len(args) >= 1 and isinstance(args[0], dict):
                rec = self._normalize_record(args[0])
                self._write_jsonl(rec); print(json.dumps(rec))
                return

            # Case: payload mistakenly passed under 'level'
            if "level" in kwargs and isinstance(kwargs["level"], dict):
                rec = self._normalize_record(kwargs["level"])
                self._write_jsonl(rec); print(json.dumps(rec))
                return

            # Normal case
            msg = args[0] if (len(args) >= 1 and isinstance(args[0], str)) else kwargs.pop("message", "")
            level = kwargs.pop("level", "INFO")
            site_id = kwargs.pop("site_id", None)
            lineage = kwargs.pop("lineage", None)
            context = kwargs.pop("context", {})
            if not isinstance(context, dict):
                context = {"value": context}
            # fold any remaining kwargs into context (e.g., path=..., error=...)
            for k, v in list(kwargs.items()):
                context[k] = v

            rec = self._normalize_record({
                "time": _now_utc_iso(),
                "level": level,
                "stage": self.name,
                "site_id": site_id,
                "lineage": lineage,
                "message": msg,
                "context": context,
            })
            self._write_jsonl(rec); print(json.dumps(rec))
        except Exception as e:
            with contextlib.suppress(Exception):
                self._write_jsonl({"time": _now_utc_iso(), "level": "ERROR", "stage": self.name,
                                   "message": "Logging failure", "context": {"error": str(e)}})

    # --------------------------- I/O writers ---------------------------

    def _record_file(self, path: Path, kind: str, extra: Optional[Dict[str, Any]] = None) -> None:
        sha = _sha256_file(path)
        rec = FileRecord(
            path=str(path.relative_to(PROJECT_ROOT).as_posix()),
            kind=kind,
            sha256=sha,
            bytes=path.stat().st_size,
            created_at=_now_utc_iso(),
            extra=extra or {},
        )
        self._manifest.append(rec)

    def _mirror(self, src: Path, dst_dir: Path) -> Path:
        _ensure_dir(dst_dir)
        dst = dst_dir / src.name
        data = src.read_bytes()
        dst.write_bytes(data)
        return dst

    def write_metric(self, name: str, data: Any) -> Tuple[Path, Optional[Path]]:
        df = _to_dataframe(data)
        # stable column order for common tidy keys
        col_order = [c for c in ("site_id", "date", "lineage") if c in df.columns]
        other_cols = [c for c in df.columns if c not in col_order]
        if col_order:
            df = df.loc[:, col_order + other_cols]

        base = _sanitize_name(name)
        csv_path = self.metrics_dir / f"{base}.csv"
        pq_path = self.metrics_dir / f"{base}.parquet"

        df.to_csv(csv_path, index=False)
        mirrored_csv = self._mirror(csv_path, self.run_metrics_dir)

        parquet_ok = None
        try:
            df.to_parquet(pq_path, index=False)
            mirrored_parq = self._mirror(pq_path, self.run_metrics_dir)
            parquet_ok = pq_path
            self._record_file(pq_path, kind="metric",
                              extra={"format": "parquet", "rows": int(df.shape[0]), "cols": int(df.shape[1])})
            self._record_file(mirrored_parq, kind="metric(run)",
                              extra={"format": "parquet", "rows": int(df.shape[0]), "cols": int(df.shape[1])})
        except Exception as e:
            self.log(level="WARNING", message=f"Parquet write unavailable for metric '{base}'", context={"name": base, "error": str(e)})

        self._record_file(csv_path, kind="metric",
                          extra={"format": "csv", "rows": int(df.shape[0]), "cols": int(df.shape[1])})
        self._record_file(mirrored_csv, kind="metric(run)",
                          extra={"format": "csv", "rows": int(df.shape[0]), "cols": int(df.shape[1])})
        return csv_path, parquet_ok

    def write_table(self, name: str, data: Any,
                    latex_caption: Optional[str] = None,
                    latex_label: Optional[str] = None) -> Tuple[Path, Path]:
        df = _to_dataframe(data)
        base = _sanitize_name(name)
        csv_path = self.tables_dir / f"{base}.csv"
        tex_path = self.tables_dir / f"{base}.tex"

        df.to_csv(csv_path, index=False)
        mirrored_csv = self._mirror(csv_path, self.run_tables_dir)

        # Try booktabs latex; fall back to minimal
        try:
            latex_str = df.to_latex(index=False, escape=True, longtable=False, bold_rows=False,
                                    caption=latex_caption, label=latex_label, na_rep="", header=True)
        except Exception:
            header = " & ".join(map(str, df.columns)) + " \\\\"
            rows = "\n".join(" & ".join(map(str, row)) + " \\\\" for row in df.itertuples(index=False))
            latex_str = "\\begin{tabular}{" + "l" * df.shape[1] + "}\n\\toprule\n" + header + "\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n"
            if latex_caption:
                latex_str = "\\begin{table}\n\\centering\n" + latex_str + f"\\caption{{{latex_caption}}}\n" + (f"\\label{{{latex_label}}}\n" if latex_label else "") + "\\end{table}\n"

        tex_path.write_text(latex_str, encoding="utf-8")
        mirrored_tex = self._mirror(tex_path, self.run_tables_dir)

        self._record_file(csv_path, kind="table",
                          extra={"format": "csv", "rows": int(df.shape[0]), "cols": int(df.shape[1])})
        self._record_file(tex_path, kind="table",
                          extra={"format": "tex", "rows": int(df.shape[0]), "cols": int(df.shape[1])})
        self._record_file(mirrored_csv, kind="table(run)",
                          extra={"format": "csv", "rows": int(df.shape[0]), "cols": int(df.shape[1])})
        self._record_file(mirrored_tex, kind="table(run)",
                          extra={"format": "tex", "rows": int(df.shape[0]), "cols": int(df.shape[1])})
        return csv_path, tex_path

    def write_figure(self, name: str, fig: Optional[matplotlib.figure.Figure] = None, dpi: int = 300) -> Tuple[Path, Path]:
        if fig is None:
            fig = plt.gcf()
        try:
            fig.tight_layout()
        except Exception:
            pass

        base = _sanitize_name(name)
        pdf_path = self.figures_dir / f"{base}.pdf"
        png_path = self.figures_dir / f"{base}.png"

        fig.savefig(pdf_path, format="pdf")
        fig.savefig(png_path, format="png", dpi=dpi)

        mirrored_pdf = self._mirror(pdf_path, self.run_figures_dir)
        mirrored_png = self._mirror(png_path, self.run_figures_dir)

        self._record_file(pdf_path, kind="figure", extra={"format": "pdf"})
        self._record_file(png_path, kind="figure", extra={"format": "png"})
        self._record_file(mirrored_pdf, kind="figure(run)", extra={"format": "pdf"})
        self._record_file(mirrored_png, kind="figure(run)", extra={"format": "png"})
        return pdf_path, png_path

    def write_report(self, text: str, append: bool = True) -> Path:
        """Write a Markdown report for the stage and mirror it to the run directory."""
        mode = "a" if (append and self.report_path.exists()) else "w"
        with self.report_path.open(mode, encoding="utf-8") as f:
            f.write(text if text.endswith("\n") else (text + "\n"))
        self._mirror(self.report_path, self.run_stage_dir)
        self._record_file(self.report_path, kind="report", extra={"format": "md", "append": append})
        return self.report_path

    def bundle_path(self) -> Path:
        return self.stage_dir / "bundle.json"

    def run_bundle_path(self) -> Path:
        return self.run_stage_dir / "bundle.json"

    def _gather_provenance(self) -> Dict[str, Any]:
        prov: Dict[str, Any] = {
            "created_at": _now_utc_iso(),
            "python": sys.version,
            "platform": os.name,
            "executable": sys.executable,
        }
        try:
            if "uprov" in globals() and hasattr(uprov, "collect_provenance"):
                prov.update(uprov.collect_provenance())  # type: ignore
        except Exception:
            pass
        return prov

    def close(self, inputs: Optional[List[str]] = None, notes: Optional[str] = None) -> None:
        if self._closed:
            return
        bundle = {
            "run_id": self.run.run_id,
            "stage": self.name,
            "created_at": self._start_time,
            "completed_at": _now_utc_iso(),
            "cfg_digest": self.run.cfg_digest,
            "manifest": [vars(m) for m in self._manifest],
            "inputs": inputs or [],
            "notes": notes or "",
            "provenance": self._gather_provenance(),
        }
        self.bundle_path().write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        self._mirror(self.bundle_path(), self.run_stage_dir)
        self._close_logs()
        self._closed = True
        self.log(level="INFO", message="Stage completed", context={"outputs": len(self._manifest)})


class RunContext:
    """
    Run-level context to manage outputs under results/ and results/runs/<run_id>/.

    Use:
        run = RunContext.start(cfg)
        ctx = run.stage("preprocessing")
        ...
        ctx.close(...)

    This class ensures:
      - Config is validated (if utils.config.validate_config is available)
      - Deterministic seeds via utils.seeds.set_global_seeds (fallback internal)
      - 'latest' pointer in results/runs (symlink if possible, else latest.txt)
    """

    def __init__(self, cfg: Mapping[str, Any], run_id: Optional[str] = None) -> None:
        self.root: Path = PROJECT_ROOT
        self.results_root: Path = self.root / "results"
        self.runs_root: Path = self.results_root / "runs"
        _ensure_dir(self.results_root)
        _ensure_dir(self.runs_root)

        # Validate config if helper exists
        self.cfg: Dict[str, Any]
        try:
            if "ucfg" in globals() and hasattr(ucfg, "validate_config"):
                self.cfg = dict(ucfg.validate_config(cfg))  # type: ignore
            else:
                self.cfg = dict(cfg)
        except Exception:
            # As a last resort, trust received cfg
            self.cfg = dict(cfg)

        self.cfg_digest: str = hashlib.sha256(json.dumps(_jsonify(self.cfg), sort_keys=True).encode("utf-8")).hexdigest()

        # Run ID
        self.run_id: str = run_id or self._make_run_id()
        self.run_root: Path = self.runs_root / self.run_id
        _ensure_dir(self.run_root)

        # Save effective config for the run
        (self.run_root / "config.json").write_text(json.dumps(_jsonify(self.cfg), indent=2), encoding="utf-8")

        # Seeds
        self.seed: int = _get_seed_from_config(self.cfg)
        _set_global_seeds_safe(self.seed)

        # Update latest pointer
        self._update_latest_pointer()

    @classmethod
    def start(cls, cfg: Mapping[str, Any], run_id: Optional[str] = None) -> "RunContext":
        """
        Start a run.

        Parameters
        ----------
        cfg : Mapping[str, Any]
            Validated configuration dictionary (or raw config; will be validated if utils.config is available).
        run_id : str, optional
            Custom run identifier.

        Returns
        -------
        RunContext
        """
        return cls(cfg=cfg, run_id=run_id)

    def stage(self, name: str) -> StageContext:
        """
        Create a StageContext for a named stage.

        Parameters
        ----------
        name : str
            Stage name (e.g., 'preprocessing', 'priors').

        Returns
        -------
        StageContext
        """
        return StageContext(self, name)

    def _make_run_id(self) -> str:
        ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        rnd = hashlib.sha256(f"{ts}-{random.random()}".encode()).hexdigest()[:8]
        return f"{ts}_{rnd}"

    def _update_latest_pointer(self) -> None:
        latest = self.runs_root / "latest"
        target = self.run_root
        # Try to create/update a symlink; if fails, write latest.txt
        try:
            if latest.exists() or latest.is_symlink():
                with contextlib.suppress(Exception):
                    latest.unlink()
            latest.symlink_to(target, target_is_directory=True)
        except Exception:
            # Fallback: write latest.txt with run_id
            (self.runs_root / "latest.txt").write_text(self.run_id, encoding="utf-8")

    def latest_run_path(self) -> Path:
        """
        Resolve the latest run path (symlink or newest by mtime fallback).

        Returns
        -------
        Path
            Path to the latest run directory.
        """
        latest = self.runs_root / "latest"
        if latest.exists() and latest.is_symlink():
            with contextlib.suppress(Exception):
                return latest.resolve()
        # Fallback to newest run by modification time
        runs = [p for p in self.runs_root.iterdir() if p.is_dir() and p.name != "latest"]
        if not runs:
            return self.run_root
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return runs[0]