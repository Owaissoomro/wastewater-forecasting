# utils/run.py  
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
from typing import Any, Dict, Mapping, Optional, Tuple, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---------- tiny helpers ----------

def _now_utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()

def _sanitize_name(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.+") else "_" for c in name]
    out = "".join(keep).strip("._")
    return out or "unnamed"

def _jsonify(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None: return x
    if isinstance(x, (np.bool_,)):      return bool(x)
    if isinstance(x, (np.integer,)):    return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x);  return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(x, (np.ndarray,)):    return x.tolist()
    if isinstance(x, pd.Timestamp):     return x.isoformat()
    if isinstance(x, pd.Timedelta):     return str(x)
    if isinstance(x, pd.Series):        return _jsonify(x.to_dict())
    if isinstance(x, pd.DataFrame):     return _jsonify(x.to_dict(orient="records"))
    if isinstance(x, dict):             return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):    return [_jsonify(v) for v in x]
    with contextlib.suppress(Exception): return json.loads(json.dumps(x))
    return str(x)

def _to_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame): return obj.copy()
    if isinstance(obj, pd.Series):    return obj.to_frame().reset_index()
    if isinstance(obj, Mapping):      return pd.DataFrame([obj])
    if isinstance(obj, (list, tuple)):
        if not obj: return pd.DataFrame()
        if isinstance(obj[0], Mapping): return pd.DataFrame(obj)
        return pd.DataFrame({"value": list(obj)})
    if np.isscalar(obj):              return pd.DataFrame({"value": [obj]})
    return pd.DataFrame(obj)

def _matplotlib_style() -> None:
    matplotlib.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "figure.autolayout": False,
    })

def _seed_from_cfg(cfg: Mapping[str, Any], default: int = 12345) -> int:
    v = cfg.get("seed")
    if isinstance(v, (int, np.integer)): return int(v)
    for k in ("run", "runtime", "experiment"):
        vv = (cfg.get(k) or {}).get("seed")
        if isinstance(vv, (int, np.integer)): return int(vv)
    return int(default)

def _set_seeds(seed: int) -> None:
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed); np.random.seed(seed)
    except Exception:
        pass

# ---------- data types ----------

@dataclass
class FileRecord:
    path: str
    kind: str
    sha256: str
    bytes: int
    created_at: str
    extra: Dict[str, Any] = field(default_factory=dict)

# ======================= StageContext ========================

class StageContext:
    """
    Minimal stage context:
      • CSV for metrics/tables
      • PNG+PDF for figures
      • Markdown reports
      • JSONL logs
      • bundle.json manifest (no provenance)
    """

    def __init__(self, run: "RunContext", name: str) -> None:
        self.run = run
        self.name = _sanitize_name(name)
        _matplotlib_style()

        # dirs (global + per-run)
        self.stage_dir       = self.run.results_root / self.name
        self.metrics_dir     = self.stage_dir / "metrics"
        self.tables_dir      = self.stage_dir / "tables"
        self.figures_dir     = self.stage_dir / "figures"
        self.logs_dir        = self.stage_dir / "logs"

        self.run_stage_dir   = self.run.runs_root / self.run.run_id / self.name
        self.run_metrics_dir = self.run_stage_dir / "metrics"
        self.run_tables_dir  = self.run_stage_dir / "tables"
        self.run_figures_dir = self.run_stage_dir / "figures"
        self.run_logs_dir    = self.run_stage_dir / "logs"

        for d in (self.stage_dir, self.metrics_dir, self.tables_dir, self.figures_dir, self.logs_dir,
                  self.run_stage_dir, self.run_metrics_dir, self.run_tables_dir, self.run_figures_dir, self.run_logs_dir):
            _ensure_dir(d)

        self.report_path = self.stage_dir / "report.md"
        self._manifest: List[FileRecord] = []
        self._start_time = _now_utc_iso()
        self._closed = False

        # logs
        self._log_fp: Optional[io.TextIOBase] = None
        self._run_log_fp: Optional[io.TextIOBase] = None
        self._open_logs()

        # per-stage deterministic seed
        stage_seed = (self.run.seed * 1315423911) ^ (hash(self.name) & 0xFFFFFFFF)
        _set_seeds(stage_seed)

        self.log(level="INFO", message="Stage started",
                 context={"run_id": self.run.run_id, "stage_seed": stage_seed, "cfg_digest": self.run.cfg_digest})

    # ----- logging -----
    def _open_logs(self) -> None:
        self._log_fp     = (self.logs_dir     / f"{self.name}.jsonl").open("a", encoding="utf-8")
        self._run_log_fp = (self.run_logs_dir / f"{self.name}.jsonl").open("a", encoding="utf-8")

    def _close_logs(self) -> None:
        with contextlib.suppress(Exception):
            if self._log_fp: self._log_fp.flush(); self._log_fp.close()
        with contextlib.suppress(Exception):
            if self._run_log_fp: self._run_log_fp.flush(); self._run_log_fp.close()
        self._log_fp = self._run_log_fp = None

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        r = dict(record)
        if isinstance(r.get("level"), dict) and not r.get("message"):
            inner = r["level"];  r = dict(inner) if isinstance(inner, dict) else r
        r.setdefault("time", _now_utc_iso())
        r.setdefault("level", "INFO")
        r.setdefault("stage", self.name)
        r.setdefault("site_id", None)
        r.setdefault("lineage", None)
        r.setdefault("message", "")
        ctx = r.get("context")
        r["context"] = dict(ctx) if isinstance(ctx, dict) else ({ } if ctx is None else {"value": ctx})
        allowed = {"time","level","stage","site_id","lineage","message","context"}
        extras = {k:v for k,v in r.items() if k not in allowed}
        if extras:
            r["context"].update(extras)
            for k in list(extras): r.pop(k, None)
        return r

    def _write_jsonl(self, payload: Dict[str, Any]) -> None:
        js = json.dumps(payload, ensure_ascii=False)
        if self._log_fp:     self._log_fp.write(js + "\n");     self._log_fp.flush()
        if self._run_log_fp: self._run_log_fp.write(js + "\n"); self._run_log_fp.flush()

    def log(self, *args, **kwargs) -> None:
        try:
            if args and isinstance(args[0], dict):
                rec = self._normalize_record(args[0]); self._write_jsonl(rec); print(json.dumps(rec)); return
            if isinstance(kwargs.get("level"), dict):
                rec = self._normalize_record(kwargs["level"]); self._write_jsonl(rec); print(json.dumps(rec)); return

            msg     = args[0] if (args and isinstance(args[0], str)) else kwargs.pop("message", "")
            level   = kwargs.pop("level", "INFO")
            site_id = kwargs.pop("site_id", None)
            lineage = kwargs.pop("lineage", None)
            context = kwargs.pop("context", {})
            if not isinstance(context, dict): context = {"value": context}
            for k, v in list(kwargs.items()): context[k] = v

            rec = self._normalize_record({
                "time": _now_utc_iso(), "level": level, "stage": self.name,
                "site_id": site_id, "lineage": lineage, "message": msg, "context": context
            })
            self._write_jsonl(rec); print(json.dumps(rec))
        except Exception as e:
            self._write_jsonl({"time": _now_utc_iso(), "level": "ERROR", "stage": self.name,
                               "message": "Logging failure", "context": {"error": str(e)}})

    # ----- I/O writers -----
    def _record_file(self, path: Path, kind: str, extra: Optional[Dict[str, Any]] = None) -> None:
        rec = FileRecord(
            path=str(path.relative_to(PROJECT_ROOT).as_posix()),
            kind=kind, sha256=_sha256_file(path), bytes=path.stat().st_size,
            created_at=_now_utc_iso(), extra=extra or {},
        )
        self._manifest.append(rec)

    def _mirror(self, src: Path, dst_dir: Path) -> Path:
        _ensure_dir(dst_dir)
        dst = dst_dir / src.name
        dst.write_bytes(src.read_bytes())
        return dst

    def write_metric(self, name: str, data: Any) -> Tuple[Path, Optional[Path]]:
        df = _to_dataframe(data)
        base = _sanitize_name(name)
        csv_path = self.metrics_dir / f"{base}.csv"
        df.to_csv(csv_path, index=False)
        self._mirror(csv_path, self.run_metrics_dir)
        self._record_file(csv_path, kind="metric", extra={"format": "csv", "rows": int(df.shape[0]), "cols": int(df.shape[1])})
        return csv_path, None

    def write_table(self, name: str, data: Any, *_args, **_kwargs) -> Tuple[Path, Optional[Path]]:
        df = _to_dataframe(data)
        base = _sanitize_name(name)
        csv_path = self.tables_dir / f"{base}.csv"
        df.to_csv(csv_path, index=False)
        self._mirror(csv_path, self.run_tables_dir)
        self._record_file(csv_path, kind="table", extra={"format": "csv", "rows": int(df.shape[0]), "cols": int(df.shape[1])})
        return csv_path, None

    def write_figure(self, name: str, fig: Optional[matplotlib.figure.Figure] = None, dpi: int = 300) -> Tuple[Path, Path]:
        fig = fig or plt.gcf()
        with contextlib.suppress(Exception): fig.tight_layout()
        base = _sanitize_name(name)
        pdf_path = self.figures_dir / f"{base}.pdf"
        png_path = self.figures_dir / f"{base}.png"
        fig.savefig(pdf_path, format="pdf")
        fig.savefig(png_path, format="png", dpi=dpi)
        self._mirror(pdf_path, self.run_figures_dir)
        self._mirror(png_path, self.run_figures_dir)
        self._record_file(pdf_path, kind="figure", extra={"format": "pdf"})
        self._record_file(png_path, kind="figure", extra={"format": "png"})
        return pdf_path, png_path

    def write_report(self, text: str, append: bool = True) -> Path:
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

    def close(self, inputs: Optional[List[str]] = None, notes: Optional[str] = None) -> None:
        if self._closed: return
        bundle = {
            "run_id": self.run.run_id, "stage": self.name,
            "created_at": self._start_time, "completed_at": _now_utc_iso(),
            "cfg_digest": self.run.cfg_digest, "manifest": [vars(m) for m in self._manifest],
            "inputs": inputs or [], "notes": notes or "",
        }
        self.bundle_path().write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        self._mirror(self.bundle_path(), self.run_stage_dir)
        self.log(level="INFO", message="Stage completed", context={"outputs": len(self._manifest)})
        self._close_logs()
        self._closed = True

# ======================= RunContext ==========================

class RunContext:
    """
    Minimal run manager:
      • results/ + results/runs/<run_id>/
      • saves config.json in run dir
      • deterministic seeding
      • 'latest.txt' pointer (no symlink)
    """

    def __init__(self, cfg: Mapping[str, Any], run_id: Optional[str] = None) -> None:
        self.root         = PROJECT_ROOT
        self.results_root = self.root / "results"
        self.runs_root    = self.results_root / "runs"
        _ensure_dir(self.results_root); _ensure_dir(self.runs_root)

        self.cfg = dict(cfg)
        self.cfg_digest = hashlib.sha256(json.dumps(_jsonify(self.cfg), sort_keys=True).encode("utf-8")).hexdigest()

        self.run_id   = run_id or self._make_run_id()
        self.run_root = self.runs_root / self.run_id
        _ensure_dir(self.run_root)
        (self.run_root / "config.json").write_text(json.dumps(_jsonify(self.cfg), indent=2), encoding="utf-8")

        self.seed = _seed_from_cfg(self.cfg)
        _set_seeds(self.seed)

        (self.runs_root / "latest.txt").write_text(self.run_id, encoding="utf-8")

    @classmethod
    def start(cls, cfg: Mapping[str, Any], run_id: Optional[str] = None) -> "RunContext":
        return cls(cfg=cfg, run_id=run_id)

    def stage(self, name: str) -> StageContext:
        return StageContext(self, name)

    def _make_run_id(self) -> str:
        ts  = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        rnd = hashlib.sha256(f"{ts}-{random.random()}".encode()).hexdigest()[:8]
        return f"{ts}_{rnd}"

    def latest_run_path(self) -> Path:
        txt = self.runs_root / "latest.txt"
        if txt.exists():
            rid = txt.read_text(encoding="utf-8").strip()
            p = self.runs_root / rid
            if p.exists(): return p
        return self.run_root