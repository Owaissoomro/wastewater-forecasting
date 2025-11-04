#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wright–Fisher Variant Forecasting — Full Pipeline Runner (single, correct version)

Stages (canonical + aliases):
  preprocessing (prep)
  priors        (prior)
  likelihood    (like, deconv)
  forecast      (fore)
  detection     (detect)
  diagnostics   (diag)
  benchmarks    (bench)
  validation    (val, validate)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# ----- repo bootstrap -----
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

import numpy as np

# ----- optional utilities -----
try:
    from utils.plotting import set_matplotlib_style  # type: ignore
except Exception:
    def set_matplotlib_style() -> None:
        pass

try:
    from utils.seeds import set_global_seeds  # type: ignore
except Exception:
    def set_global_seeds(seed: int) -> None:
        import os, random
        os.environ["PYTHONHASHSEED"] = str(int(seed))
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass

# project config helpers
try:
    from utils.config import load_config as utils_load_config  # type: ignore
except Exception:
    utils_load_config = None  # type: ignore

try:
    from utils.config import validate_config as utils_validate_config  # type: ignore
except Exception:
    utils_validate_config = None  # type: ignore

# run context
from utils.run import RunContext  # type: ignore

# stage runners (each must accept (cfg, ctx))
from stages.preprocessing import run_preprocessing  # type: ignore
from stages.priors import run_priors  # type: ignore
from stages.likelihood import run_likelihood  # type: ignore
from stages.forecast import run_forecast  # type: ignore


# ===== CONFIG LOADING & VALIDATION =====

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _yaml_load(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a mapping: {path}")
    return data


def _jsonschema_validate(cfg: Dict[str, Any], schema_path: Path) -> None:
    import json
    from jsonschema import Draft7Validator, RefResolver
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    resolver = RefResolver(base_uri=str(schema_path.resolve().as_uri()), referrer=schema)  # type: ignore[arg-type]
    validator = Draft7Validator(schema, resolver=resolver)
    errors = sorted(validator.iter_errors(cfg), key=lambda e: e.path)
    if errors:
        msgs = []
        for e in errors:
            loc = ".".join(str(x) for x in e.absolute_path)
            msgs.append(f"{loc}: {e.message}")
        raise ValueError("Config validation failed:\n" + "\n".join(msgs))


def load_and_validate_config(paths: List[Path], schema_path: Path) -> Dict[str, Any]:
    """
    Load & merge configs (later files override earlier), then validate.

    Your utils.config.load_config expects a single path (or mapping), *not* a list.
    We call it once per file and deep-merge results; then validate with
    utils.config.validate_config if available, else JSON Schema.
    """
    # resolve relative paths to repo root
    resolved: List[Path] = []
    for p in paths:
        pp = p if p.is_absolute() else (_REPO_ROOT / p).resolve()
        if not pp.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        resolved.append(pp)

    # try project loader file-by-file
    cfg: Dict[str, Any] = {}
    used_project_loader = False
    if utils_load_config is not None:
        try:
            tmp: Dict[str, Any] = {}
            for p in resolved:
                piece = utils_load_config(str(p))  # <-- single path (string)
                if not isinstance(piece, dict):
                    raise TypeError(f"utils.config.load_config returned non-mapping for {p}")
                tmp = _deep_merge(tmp, piece)
            cfg = tmp
            used_project_loader = True
        except Exception:
            used_project_loader = False

    if not used_project_loader:
        # manual merge via YAML
        tmp: Dict[str, Any] = {}
        for p in resolved:
            part = _yaml_load(p)
            tmp = _deep_merge(tmp, part)
        cfg = tmp

    # validate
    validated = False
    if utils_validate_config is not None:
        try:
            utils_validate_config(cfg, schema_path=str(schema_path))  # most repos accept this
            validated = True
        except TypeError:
            # some older validators accept (cfg, schema_path) positional
            utils_validate_config(cfg, str(schema_path))  # type: ignore
            validated = True
        except Exception:
            validated = False

    if not validated:
        _jsonschema_validate(cfg, schema_path)

    return cfg


# ===== STAGE MAPPING / SELECTION =====

def _normalize_stage_name(name: str) -> str:
    key = (name or "").strip().lower()
    aliases = {
        "preprocessing": "preprocessing", "prep": "preprocessing",
        "priors": "priors", "prior": "priors",
        "likelihood": "likelihood", "like": "likelihood", "deconv": "likelihood",
        "forecast": "forecast", "fore": "forecast",
        "detection": "detection", "detect": "detection",
        "diagnostics": "diagnostics", "diag": "diagnostics",
        "benchmarks": "benchmarks", "bench": "benchmarks",
        "validation": "validation", "validate": "validation", "val": "validation",
    }
    if key not in aliases:
        raise ValueError(f"Unknown stage: {name}")
    return aliases[key]


def _get_stage_funcs() -> Dict[str, Callable[[Dict[str, Any], Any], Any]]:
    return {
        "preprocessing": run_preprocessing,
        "priors":        run_priors,
        "likelihood":    run_likelihood,
        "forecast":      run_forecast,
    }


def _stage_order() -> List[str]:
    return ["preprocessing", "priors", "likelihood"]

def _parse_stage_selection(arg: Optional[str]) -> List[str]:
    if arg is None or arg.strip().lower() in {"", "all"}:
        return _stage_order()
    names = [s for s in (x.strip() for x in arg.split(",")) if s]
    normalized = [_normalize_stage_name(n) for n in names]
    seen, ordered = set(), []
    for n in normalized:
        if n not in seen:
            ordered.append(n); seen.add(n)
    return ordered


# ===== CTX HELPERS =====

def _ctx_stage_dir(ctx: Any, stage: str, repo_root: Path) -> Path:
    for attr in ("stage_dir", "out_dir", "base_dir", "dir", "outpath"):
        if hasattr(ctx, attr):
            p = getattr(ctx, attr)
            if isinstance(p, (str, Path)):
                return Path(p)
    return repo_root / "results" / stage


def _ctx_log(ctx: Any, level: str, stage: str, message: str,
             site_id: Optional[str] = None, lineage: Optional[str] = None,
             context: Optional[Dict[str, Any]] = None) -> None:
    record = {
        "time": str(np.datetime64("now").astype("datetime64[ms]")),
        "level": level.upper(),
        "stage": stage,
        "site_id": site_id,
        "lineage": lineage,
        "message": message,
        "context": context or {},
    }
    for meth in ("write_log", "log_json", "log"):
        fn = getattr(ctx, meth, None)
        if callable(fn):
            try:
                fn(record)
                return
            except Exception:
                continue
    # no-op if no logging method


# ===== ORCHESTRATION =====

def run_pipeline(config_paths: List[Path], stages: List[str], seed_override: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    repo_root = _REPO_ROOT
    schema_path = repo_root / "configs" / "schema.json"
    cfg = load_and_validate_config([Path(p) for p in config_paths], Path(schema_path))

    # seed (prefer run.seed)
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    seed = int(seed_override if seed_override is not None else run_cfg.get("seed", 12345))
    set_global_seeds(seed)

    set_matplotlib_style()

    run = RunContext.start(cfg)

    stage_funcs = _get_stage_funcs()
    stage_dirs: Dict[str, Path] = {}

    for stage in stages:
        if stage not in stage_funcs:
            raise ValueError(f"Stage runner not found for: {stage}")

        fn = stage_funcs[stage]
        ctx = run.stage(stage)
        _ctx_log(ctx, "INFO", stage, f"Starting stage '{stage}'", context={"seed": seed})

        try:
            fn(cfg, ctx)
            _ctx_log(ctx, "INFO", stage, f"Completed stage '{stage}'")
        except Exception as exc:
            _ctx_log(ctx, "ERROR", stage, f"Stage '{stage}' failed", context={"error": str(exc)})
            try:
                if hasattr(run, "close"):
                    run.close(inputs=[str(p) for p in config_paths], notes=f"FAILED at stage '{stage}'")
            except Exception:
                pass
            raise

        stage_dirs[stage] = _ctx_stage_dir(ctx, stage, repo_root)

    try:
        if hasattr(run, "close"):
            run.close(inputs=[str(p) for p in config_paths], notes="pipeline completed")
    except Exception:
        pass

    return cfg, stage_dirs


# ===== CLI =====

def _default_config_paths(repo_root: Path) -> List[Path]:
    return [repo_root / "configs" / "default.yaml"]

def _print_summary(stage_dirs: Dict[str, Path]) -> None:
    print("Pipeline completed. Stage outputs:")
    for stage, p in stage_dirs.items():
        print(f"  - {stage}: {p}")

def main(argv: Optional[Iterable[str]] = None) -> None:
    repo_root = _REPO_ROOT
    parser = argparse.ArgumentParser(description="Wright–Fisher Variant Forecasting — Pipeline Runner")
    parser.add_argument("-c", "--config", dest="configs", action="append", default=None,
                        help="Path to YAML config (repeatable). Defaults to configs/default.yaml under repo root.")
    parser.add_argument("--stages", type=str, default="all",
                        help=("Comma-separated list or 'all'. Valid: preprocessing,priors,likelihood,forecast,"
                              "detection,diagnostics,benchmarks,validation. Aliases supported."))
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from config.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg_paths = [Path(p) for p in args.configs] if args.configs else _default_config_paths(repo_root)
    cfg_paths = [p if p.is_absolute() else (repo_root / p).resolve() for p in cfg_paths]

    stages = _parse_stage_selection(args.stages)

    try:
        _, stage_dirs = run_pipeline(config_paths=cfg_paths, stages=stages, seed_override=args.seed)
    except Exception as exc:
        sys.stderr.write(f"Pipeline failed: {exc}\n")
        sys.exit(2)

    _print_summary(stage_dirs)

if __name__ == "__main__":
    main()
