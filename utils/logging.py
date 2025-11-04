from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

try:
    import numpy as np
except Exception:  # numpy is optional
    np = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pandas is optional
    pd = None  # type: ignore


PathLike = Union[str, os.PathLike, Path]


def _iso_utc(ts: float) -> str:
    """Format POSIX timestamp as ISO 8601 UTC with Z suffix."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    s = dt.isoformat()
    return s[:-6] + "Z" if s.endswith("+00:00") else s


def _json_safe(obj: Any, max_ndarray_elems: int = 1024) -> Any:
    """Recursively convert obj to a JSON-serializable structure."""
    # Primitives / None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Python date/time
    if isinstance(obj, (datetime, date)):
        if isinstance(obj, datetime) and obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.isoformat()

    # Paths
    if isinstance(obj, (Path, os.PathLike)):
        return str(obj)

    # Numpy
    if np is not None:
        if isinstance(obj, np.generic):  # numpy scalar
            try:
                return obj.item()
            except Exception:
                pass
        if isinstance(obj, np.ndarray):
            size = int(obj.size)
            if size <= max_ndarray_elems:
                return obj.tolist()
            return {"ndarray": True, "shape": list(obj.shape), "dtype": str(obj.dtype), "summary": "truncated"}

    # Pandas
    if pd is not None:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Timedelta):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, pd.DataFrame):
            return {
                "columns": [str(c) for c in obj.columns],
                "data": obj.astype(object).applymap(_json_safe).values.tolist(),
            }
        if isinstance(obj, pd.Index):
            return [_json_safe(v) for v in obj.tolist()]
        if obj is getattr(pd, "NaT", object()):
            return None

    # Mappings
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    # Iterables
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]

    # Fallback
    try:
        return str(obj)
    except Exception:
        return repr(obj)


class JsonLineFormatter(logging.Formatter):
    """Structured JSONL formatter: {time, level, stage, site_id, lineage, message, context}."""

    def format(self, record: logging.LogRecord) -> str:
        stage = getattr(record, "stage", None)
        site_id = getattr(record, "site_id", None)
        lineage = getattr(record, "lineage", None)

        ctx = getattr(record, "context", {})
        if not isinstance(ctx, Mapping):
            ctx = {"value": ctx}

        # Include exception info if present
        if record.exc_info:
            exc_type, exc_obj, _ = record.exc_info
            ctx = dict(ctx)
            ctx["exception"] = {
                "type": getattr(exc_type, "__name__", str(exc_type)),
                "message": str(exc_obj),
            }

        payload = {
            "time": _iso_utc(record.created),
            "level": record.levelname,
            "stage": stage,
            "site_id": site_id,
            "lineage": lineage,
            "message": record.getMessage(),
            "context": _json_safe(ctx),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class BoundLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that binds default fields for structured JSON logging."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        stage: Optional[str] = None,
        site_id: Optional[str] = None,
        lineage: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(logger, {})
        self._bound = {
            "stage": stage,
            "site_id": site_id,
            "lineage": lineage,
            "context": dict(context) if isinstance(context, Mapping) else {},
        }

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        extra = kwargs.pop("extra", {}) or {}
        if not isinstance(extra, dict):
            extra = {"extra": extra}

        # Merge scalar fields with call priority
        stage = extra.get("stage", self._bound.get("stage"))
        site_id = extra.get("site_id", self._bound.get("site_id"))
        lineage = extra.get("lineage", self._bound.get("lineage"))

        # Merge contexts
        call_ctx = extra.get("context", {})
        base_ctx = self._bound.get("context", {})
        if not isinstance(call_ctx, Mapping):
            call_ctx = {"value": call_ctx}
        merged_ctx = dict(base_ctx)
        merged_ctx.update(call_ctx)

        extra_out = dict(extra)
        extra_out.update({"stage": stage, "site_id": site_id, "lineage": lineage, "context": merged_ctx})
        kwargs["extra"] = extra_out
        return msg, kwargs

    def bind(
        self,
        *,
        stage: Optional[str] = None,
        site_id: Optional[str] = None,
        lineage: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> "BoundLoggerAdapter":
        new_ctx = dict(self._bound.get("context") or {})
        if isinstance(context, Mapping):
            new_ctx.update(context)
        return BoundLoggerAdapter(
            self.logger,
            stage=stage or self._bound.get("stage"),
            site_id=site_id or self._bound.get("site_id"),
            lineage=lineage or self._bound.get("lineage"),
            context=new_ctx,
        )


# --- Time helpers -------------------------------------------------------------

def utcnow_iso(tz: str = "UTC", milliseconds: bool = True) -> str:
    """
    Return an ISO-8601 timestamp string for 'now'.
    Default: UTC ISO, with milliseconds. If an IANA timezone is given and available, convert.
    """
    dt = datetime.now(timezone.utc)
    try:
        if tz and tz != "UTC":
            from zoneinfo import ZoneInfo  # Python 3.9+
            dt = dt.astimezone(ZoneInfo(tz))
    except Exception:
        pass
    timespec = "milliseconds" if milliseconds else "seconds"
    return dt.isoformat(timespec=timespec)


# --- Logger setup -------------------------------------------------------------

def init_json_logger(name: str, log_file: PathLike) -> logging.Logger:
    """Init a JSONL logger (one JSON object per line). Idempotent per (name, path)."""
    path = Path(log_file).expanduser()
    try:
        path = path.resolve()  # keep absolute, even if file doesn't exist yet
    except Exception:
        path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    target_path_str = str(path)
    for h in logger.handlers:
        if getattr(h, "_jsonl_handler", False) and getattr(h, "_jsonl_path", None) == target_path_str:
            return logger

    fh = logging.FileHandler(target_path_str, mode="a", encoding="utf-8", delay=True)
    fh.setLevel(logging.INFO)
    fh.setFormatter(JsonLineFormatter())
    fh._jsonl_handler = True  # type: ignore[attr-defined]
    fh._jsonl_path = target_path_str  # type: ignore[attr-defined]
    logger.addHandler(fh)
    return logger


def bind_logger(
    logger: logging.Logger,
    *,
    stage: Optional[str] = None,
    site_id: Optional[str] = None,
    lineage: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> BoundLoggerAdapter:
    """Return a BoundLoggerAdapter with defaults for stage/site_id/lineage/context."""
    return BoundLoggerAdapter(logger, stage=stage, site_id=site_id, lineage=lineage, context=context)


def log_event(
    logger: Union[logging.Logger, BoundLoggerAdapter],
    level: Union[int, str],
    *,
    stage: Optional[str],
    site_id: Optional[str],
    lineage: Optional[str],
    message: str,
    context: Optional[Mapping[str, Any]] = None,
) -> None:
    """Emit a structured log event."""
    if isinstance(level, str):
        levelno = getattr(logging, level.upper(), logging.INFO)
    else:
        levelno = int(level)
    extra = {"stage": stage, "site_id": site_id, "lineage": lineage, "context": context or {}}
    logger.log(levelno, message, extra=extra)


__all__ = [
    "init_json_logger",
    "JsonLineFormatter",
    "BoundLoggerAdapter",
    "bind_logger",
    "log_event",
    "utcnow_iso",
]
