import re
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, spearmanr, pearsonr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator

from utils.run import RunContext
from utils.plotting import set_matplotlib_style, place_legend_below, categorical_palette, continuous_cmap

# Optional seaborn (palettes); gracefully fall back if not present
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False
    warnings.warn("seaborn not found – using matplotlib fallbacks for styling.")

try:
    from utils.seeds import set_global_seeds
except Exception:
    def set_global_seeds(seed: int) -> None:
        import random, os
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------
# Config
# ---------------------------

@dataclass
class PreprocConfig:
    min_coverage: int
    left_censor_af: float
    min_alt_count: int
    bias_dropout_quantile: float
    bias_min_highcov_quantile: float
    bias_af_pos_rate_max: float
    ridgeline_sites_max: int
    heatmap_mutations_max: int
    seed: int


# ---------------------------
# Utilities
# ---------------------------

def _canonicalize_mutation(mut: str) -> str:
    """
    Canonicalize a mutation identifier into a strict, uppercase, delimiter-tolerant form.

    This ensures that mutation labels are consistent across input files, plots,
    and feature tables by:
      • Stripping whitespace
      • Removing arrows ('->')
      • Keeping only alphanumeric characters and delimiters (:, _, -)
      • Converting to uppercase

    Parameters
    ----------
    mut : str
        Original mutation identifier (may contain mixed case, spaces, etc.)

    Returns
    -------
    str
        Canonicalized mutation ID (e.g., "aa:e:p71l" → "AA:E:P71L").
    """
    if pd.isna(mut):
        return ""
    s = str(mut).strip().upper().replace(" ", "").replace("->", "")
    s = re.sub(r"[^A-Z0-9:_\-]", "", s)
    return s


def _palette_from_cmap(n: int, cmap_name: str) -> List:
    """
    Generate an evenly spaced color palette from a named Matplotlib colormap.

    Parameters
    ----------
    n : int
        Number of discrete colors to generate.
    cmap_name : str
        Name of the Matplotlib colormap (e.g., 'viridis', 'tab10', 'Set2').

    Returns
    -------
    List[tuple]
        List of RGBA tuples representing sampled colors.
    """
    cmap = plt.cm.get_cmap(cmap_name, max(n, 1))
    return [cmap(i) for i in range(max(n, 1))]


def _build_global_color_maps(
    all_mutations: List[str],
    all_lineages: List[str],
) -> Tuple[Dict[str, tuple], Dict[str, tuple]]:
    """
    Build global color maps for mutations and lineages, ensuring consistent visual identity
    across all figures in the report.

    Preference order:
      1. Seaborn palettes (if available): 'Set2' for mutations, 'Dark2' for lineages.
      2. Matplotlib fallback colormaps ('Set2', 'Dark2', or their tab alternatives).

    Parameters
    ----------
    all_mutations : List[str]
        List of unique mutation identifiers.
    all_lineages : List[str]
        List of unique lineage identifiers.

    Returns
    -------
    (Dict[str, tuple], Dict[str, tuple])
        Two dictionaries mapping each mutation and lineage to an RGBA color tuple.

    Notes
    -----
    Colors are assigned deterministically by sorted order of identifiers, ensuring
    reproducibility across runs.
    """
    if HAS_SEABORN:
        mut_colors = sns.color_palette("Set2", n_colors=max(8, len(all_mutations)))
        lin_colors = sns.color_palette("Dark2", n_colors=max(8, len(all_lineages)))
    else:
        mut_colors = _palette_from_cmap(
            max(8, len(all_mutations)),
            "Set2" if "Set2" in plt.colormaps() else "tab20"
        )
        lin_colors = _palette_from_cmap(
            max(8, len(all_lineages)),
            "Dark2" if "Dark2" in plt.colormaps() else "tab10"
        )

    mut_to_color = {m: mut_colors[i % len(mut_colors)] for i, m in enumerate(sorted(all_mutations))}
    lin_to_color = {l: lin_colors[i % len(lin_colors)] for i, l in enumerate(sorted(all_lineages))}

    return mut_to_color, lin_to_color

# ---------------------------
# Strict loader & coercion 
# ---------------------------
def _load_and_coerce_long(path: str) -> pd.DataFrame:
    """
    Load a Jahn-like long SNV table and coerce it into a strict, predictable schema.
    """
    df = pd.read_csv(path)

    # --- Column presence check ---
    need = ["sample_id", "site_id", "date", "mutation", "count", "coverage"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Dtypes (IDs as string) ---
    df["sample_id"] = df["sample_id"].astype("string").fillna("NA")
    df["site_id"]   = df["site_id"].astype("string")
    df["mutation"]  = df["mutation"].astype("string")

    # Canonicalize mutation early (uppercased, restricted charset)
    df["mutation"] = df["mutation"].map(_canonicalize_mutation)

    # --- Date parsing: strict YYYY-MM-DD, drop invalid, normalize to 00:00 ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="%Y-%m-%d")
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.normalize()

    # --- Numeric coercion: non-negative int64; cap count at coverage ---
    for c in ("count", "coverage"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
        df.loc[df[c] < 0, c] = 0
    df.loc[df["count"] > df["coverage"], "count"] = df["coverage"]

    # --- Deduplicate by summing counts/coverage over the key ---
    key = ["site_id", "date", "sample_id", "mutation"]
    df = (
        df.groupby(key, observed=True, dropna=False)[["count", "coverage"]]
          .sum()
          .reset_index()
    )

    # --- Stable sort for reproducibility ---
    df = (
        df.sort_values(["site_id", "date", "sample_id", "mutation"], kind="mergesort")
          .reset_index(drop=True)
    )
    return df


def _validate_jahn_like_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Validate the Jahn-like long SNV table after strict loading/coercion.
    """
    validations: List[Dict] = []
    required_cols = ["sample_id", "site_id", "date", "mutation", "count", "coverage"]

    # Column presence
    for col in required_cols:
        ok = col in df.columns
        validations.append({"table": "jahn_like", "check": f"has_column:{col}", "passed": bool(ok), "details": ""})
    if not all(v["passed"] for v in validations[-len(required_cols):]):
        return df, validations

    # Date type and normalization
    ok_date_type = pd.api.types.is_datetime64_any_dtype(df["date"])
    validations.append({"table": "jahn_like", "check": "date_is_datetime64", "passed": bool(ok_date_type), "details": f"dtype={df['date'].dtype}"})
    ok_date_norm = bool((df["date"] == df["date"].dt.normalize()).all()) if ok_date_type else False
    validations.append({"table": "jahn_like", "check": "date_is_normalized_daily", "passed": ok_date_norm, "details": ""})

    # String-like columns (type + value-level sanity)
    for col in ["sample_id", "site_id", "mutation"]:
        ok_type = pd.api.types.is_string_dtype(df[col])
        if ok_type:
            mask_invalid = (~df[col].isna()) & (~df[col].map(lambda x: isinstance(x, str)))
            invalid_val_count = int(mask_invalid.sum())
        else:
            invalid_val_count = int((~df[col].isna()).sum())
        validations.append({"table": "jahn_like", "check": f"{col}_is_string_dtype", "passed": bool(ok_type), "details": f"dtype={df[col].dtype}"})
        validations.append({"table": "jahn_like", "check": f"{col}_values_are_str_or_na", "passed": invalid_val_count == 0, "details": f"invalid_count={invalid_val_count}"})

    # Numeric + nonnegative + logical constraint (count <= coverage)
    for col in ["count", "coverage"]:
        ok_num = np.issubdtype(df[col].dtype, np.number)
        ok_nonneg = bool((df[col] >= 0).all()) if ok_num else False
        validations.append({"table": "jahn_like", "check": f"{col}_is_numeric", "passed": bool(ok_num), "details": f"dtype={df[col].dtype}"})
        validations.append({"table": "jahn_like", "check": f"{col}_nonnegative", "passed": ok_nonneg, "details": f"neg_count={int((df[col] < 0).sum())}"})

    ok_le = bool((df["count"] <= df["coverage"]).all())
    violations = int((df["count"] > df["coverage"]).sum())
    validations.append({"table": "jahn_like", "check": "count_le_coverage", "passed": ok_le, "details": f"violations={violations}"})

    # Canonical mutation charset
    pattern_ok = df["mutation"].fillna("").map(lambda s: re.fullmatch(r"[A-Z0-9:_\-]*", s) is not None).all()
    validations.append({"table": "jahn_like", "check": "mutation_char_set_ok", "passed": bool(pattern_ok), "details": "allowed=[A-Z0-9:_-]"})

    # Key uniqueness
    key = ["site_id", "date", "sample_id", "mutation"]
    dup_count = int(df.duplicated(key).sum())
    validations.append({"table": "jahn_like", "check": "key_unique_after_dedup", "passed": dup_count == 0, "details": f"duplicates={dup_count}"})

    # Non-empty
    validations.append({"table": "jahn_like", "check": "non_empty_table", "passed": int(len(df)) > 0, "details": f"rows={int(len(df))}"})
    return df, validations


def _validate_signatures_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Validate the lineage signature table.
    """
    validations: List[Dict] = []
    required_cols = ["mutation", "lineage", "weight"]

    for col in required_cols:
        ok = col in df.columns
        validations.append({"table": "signatures", "check": f"has_column:{col}", "passed": bool(ok), "details": ""})
    if not all(v["passed"] for v in validations[-len(required_cols):]):
        return df, validations

    for col in ["mutation", "lineage"]:
        ok_type = pd.api.types.is_string_dtype(df[col])
        if ok_type:
            mask_invalid = (~df[col].isna()) & (~df[col].map(lambda x: isinstance(x, str)))
            invalid_val_count = int(mask_invalid.sum())
        else:
            invalid_val_count = int((~df[col].isna()).sum())
        validations.append({"table": "signatures", "check": f"{col}_is_string_dtype", "passed": bool(ok_type), "details": f"dtype={df[col].dtype}"})
        validations.append({"table": "signatures", "check": f"{col}_values_are_str_or_na", "passed": invalid_val_count == 0, "details": f"invalid_count={invalid_val_count}"})

    ok_num = np.issubdtype(df["weight"].dtype, np.number)
    validations.append({"table": "signatures", "check": "weight_is_numeric", "passed": bool(ok_num), "details": f"dtype={df['weight'].dtype}"})
    if ok_num:
        ok_range = bool(((df["weight"] >= 0) & (df["weight"] <= 1)).all())
        out_of_range = int(((df["weight"] < 0) | (df["weight"] > 1)).sum())
    else:
        ok_range = False
        out_of_range = int(len(df))
    validations.append({"table": "signatures", "check": "weight_in_[0,1]", "passed": ok_range, "details": f"out_of_range={out_of_range}"})

    validations.append({"table": "signatures", "check": "non_empty_table", "passed": int(len(df)) > 0, "details": f"rows={int(len(df))}"})
    return df, validations


def _validate_lineages_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Validate the optional lineage metadata table.
    """
    validations: List[Dict] = []
    required_cols = ["lineage", "label", "is_voc", "notes"]

    for col in required_cols:
        ok = col in df.columns
        validations.append({"table": "lineages", "check": f"has_column:{col}", "passed": bool(ok), "details": ""})
    if not all(v["passed"] for v in validations[-len(required_cols):]):
        return df, validations

    for col in ["lineage", "label", "notes"]:
        ok_type = pd.api.types.is_string_dtype(df[col])
        if ok_type:
            mask_invalid = (~df[col].isna()) & (~df[col].map(lambda x: isinstance(x, str)))
            invalid_val_count = int(mask_invalid.sum())
        else:
            invalid_val_count = int((~df[col].isna()).sum())
        validations.append({"table": "lineages", "check": f"{col}_is_string_dtype", "passed": bool(ok_type), "details": f"dtype={df[col].dtype}"})
        validations.append({"table": "lineages", "check": f"{col}_values_are_str_or_na", "passed": invalid_val_count == 0, "details": f"invalid_count={invalid_val_count}"})

    # is_voc boolean or coercible
    if df["is_voc"].dtype != bool:
        mapping = {"true": True, "false": False}
        df = df.copy()
        df["is_voc"] = df["is_voc"].astype(str).str.strip().str.lower().map(mapping)

    ok_isvoc = bool(df["is_voc"].isin([True, False]).all())
    n_na = int(df["is_voc"].isna().sum())
    validations.append({"table": "lineages", "check": "is_voc_bool_or_coercible", "passed": ok_isvoc, "details": f"na_after_coercion={n_na}"})

    validations.append({"table": "lineages", "check": "non_empty_table", "passed": int(len(df)) > 0, "details": f"rows={int(len(df))}"})
    return df, validations

# ---------------------------
# Core transforms (AF/LOD/censoring/missingness/bias)
# ---------------------------
def compute_af_censor_filter(
    df: pd.DataFrame,
    min_coverage: int,
    left_censor_af: float,
    min_alt_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute allele frequency (AF) and per-row limit-of-detection (LOD) thresholds
    in **KEEP-ALL** mode, i.e., no rows are removed and no AF values are censored.

    Definitions
    -----------
    AF (observed) :
        af = count / coverage  (with coverage==0 → af=0). Values are clipped to [0, 1].

    LOD threshold (reported only) :
        lod_threshold = max(left_censor_af, min_alt_count / coverage) for coverage > 0,
                        otherwise left_censor_af.
        This is *not* used to modify af or drop observations in KEEP-ALL mode, but is
        reported for downstream awareness/diagnostics.

    Outputs
    -------
    df_out : pandas.DataFrame
        A copy of the input with added columns:
          • 'af'            : float in [0, 1]
          • 'lod_threshold' : float ≥ 0
          • 'af_obs'        : float (identical to 'af' in KEEP-ALL mode)
          • 'af_censored'   : bool (always False in KEEP-ALL mode)
        Also sets df_out.attrs['filter_stats'] describing thresholds and row counts.

    lod_summary : pandas.DataFrame
        Per (site_id, date) summary with:
          • n_rows, n_censored (always 0 here), frac_censored (always 0)
          • mean_lod
          • coverage distribution stats (n_obs, mean, median, p05, p95, min, max)

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format SNV table (already coerced/validated upstream) with:
        ['site_id','date','sample_id','mutation','count','coverage'].
    min_coverage : int
        Minimum coverage threshold (reported for transparency; not applied here).
    left_censor_af : float
        Left-censoring AF used to form the LOD threshold (reporting only).
    min_alt_count : int
        Minimum alternate count used in LOD = max(left_censor_af, min_alt_count/coverage).

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        (df_out, lod_summary)

    Notes
    -----
    • This function intentionally does **not** censor or drop any observations; it
      reports thresholds and keeps AF exactly as observed. Downstream stages that
      need censoring should apply it explicitly.
    """
    df = df.copy()

    # AF = alt / coverage; 0 when coverage==0; clip to [0, 1]
    df["af"] = np.where(
        df["coverage"] > 0,
        df["count"] / np.maximum(df["coverage"], 1.0),
        0.0,
    ).clip(0.0, 1.0)

    # Per-row LOD threshold (reporting only; KEEP-ALL)
    df["lod_threshold"] = np.maximum(
        float(left_censor_af),
        np.where(
            df["coverage"] > 0,
            float(min_alt_count) / np.maximum(df["coverage"], 1.0),
            float(left_censor_af),
        ),
    )

    # KEEP-ALL: observed AF is retained exactly; nothing is censored
    df["af_obs"] = df["af"]
    df["af_censored"] = False

    # Coverage summary (for info only)
    coverage_summary = (
        df.groupby(["site_id", "date"])["coverage"]
          .agg(
              n_obs="size",
              mean="mean",
              median="median",
              p05=lambda x: float(np.quantile(x, 0.05)),
              p95=lambda x: float(np.quantile(x, 0.95)),
              min="min",
              max="max",
          )
          .reset_index()
          .sort_values(["site_id", "date"])
    )

    # LOD summary (for info only; nothing dropped or censored here)
    lod_summary = (
        df.groupby(["site_id", "date"])
          .agg(
              n_rows=("mutation", "size"),
              n_censored=("af_censored", "sum"),
              frac_censored=("af_censored", "mean"),
              mean_lod=("lod_threshold", "mean"),
          )
          .reset_index()
          .sort_values(["site_id", "date"])
    )
    lod_summary = lod_summary.merge(coverage_summary, on=["site_id", "date"], how="left")

    # Filter stats (explicitly documenting KEEP-ALL)
    df.attrs["filter_stats"] = {
        "min_coverage": int(min_coverage),
        "left_censor_af": float(left_censor_af),
        "min_alt_count": int(min_alt_count),
        "rows_before": int(len(df)),
        "rows_after": int(len(df)),
        "dropped_rows": 0,
    }

    return df, lod_summary

def compute_missingness(
    df: pd.DataFrame,
    signatures: Optional[pd.DataFrame],
    min_coverage: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-(site, date) missingness summary and a site×mutation heatmap
    in **KEEP-ALL** mode (no censoring or drops).

    Behavior
    --------
    • This function **does not declare any missingness**. It returns:
        - A per-(site_id, date) summary with:
            * total_mutations : number of unique mutations observed that day
            * missing         : 0
            * frac_missing    : 0.0
        - A site×mutation heatmap filled with zeros (float), where:
            * rows  = all unique sites present in `df` (sorted as strings)
            * cols  = all unique mutations from `signatures['mutation']` if provided
                      (preferred for consistent reporting), else from `df['mutation']`
            * values = 0.0 for every cell (i.e., no missingness flagged)

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format SNV table after loading/coercion with columns at least:
        ['site_id', 'date', 'mutation', 'count', 'coverage'].
        Assumes 'date' is datetime64[ns] normalized to daily.
    signatures : Optional[pandas.DataFrame]
        Lineage signature table with at least ['mutation'] if present.
        If provided, its unique mutation set defines heatmap columns to ensure
        consistent plotting across sites/dates even when some mutations do not
        appear in `df` for certain days.
    min_coverage : int
        Unused in KEEP-ALL mode; included for signature compatibility and future
        extensions where missingness may depend on coverage thresholds.

    Returns
    -------
    miss_summary : pandas.DataFrame
        Columns: ['site_id', 'date', 'total_mutations', 'missing', 'frac_missing'],
        sorted by ['site_id', 'date'].
    miss_heat : pandas.DataFrame
        A float-valued DataFrame indexed by site_id (rows) and mutation (columns),
        filled with 0.0.

    Notes
    -----
    • This is intentionally conservative: it provides a consistent reporting scaffold
      without asserting missingness. If you later decide to compute true missingness
      relative to a coverage rule, you can switch this implementation to mark cells
      where coverage < min_coverage as missing.
    """
    # Per-(site, date) summary — count unique mutations, declare no missingness
    miss_summary = (
        df.groupby(["site_id", "date"])["mutation"]
          .nunique()
          .reset_index()
          .rename(columns={"mutation": "total_mutations"})
          .sort_values(["site_id", "date"])
    )
    miss_summary["missing"] = 0
    miss_summary["frac_missing"] = 0.0

    # Heatmap scaffold: rows = sites; cols = mutations (from signatures if available)
    sites = sorted(df["site_id"].astype(str).unique().tolist())
    if signatures is not None and "mutation" in signatures.columns:
        mutations = sorted(signatures["mutation"].astype(str).unique().tolist())
    else:
        mutations = sorted(df["mutation"].astype(str).unique().tolist())

    # All zeros (float) → KEEP-ALL mode
    miss_heat = pd.DataFrame(0.0, index=sites, columns=mutations)

    return miss_summary, miss_heat

def compute_bias_loci(
    df: pd.DataFrame,
    min_coverage: int,
    left_censor_af: float,
    bias_dropout_quantile: float,
    bias_min_highcov_quantile: float,
    bias_af_pos_rate_max: float,
) -> pd.DataFrame:
    """
    Compute bias diagnostics per mutation:
      - median_coverage, coverage_ratio_to_global
      - af_pos_rate_highcov (P(AF > left_censor_af | coverage >= highcov_threshold))
      - flag_dropout, flag_ref_bias
      - thresholds: dropout_threshold, highcov_threshold, ref_bias_threshold, global_median_coverage
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "mutation", "median_coverage", "coverage_ratio_to_global",
                "af_pos_rate_highcov", "flag_dropout", "flag_ref_bias",
                "dropout_threshold", "highcov_threshold", "ref_bias_threshold",
                "global_median_coverage",
            ]
        )

    # Ensure AF is available
    if "af" not in df.columns:
        df = df.copy()
        df["af"] = np.where(df["coverage"] > 0,
                            df.get("count", 0) / np.maximum(df["coverage"], 1.0),
                            0.0).clip(0.0, 1.0)

    # Per-mutation median coverage & global ref
    cov_by_mut = df.groupby("mutation")["coverage"].median().rename("median_coverage")
    global_cov = float(df["coverage"].median()) if len(df) else 0.0
    coverage_ratio = (cov_by_mut / max(global_cov, 1.0)).rename("coverage_ratio_to_global")

    # Dropout cutoff on non-zero medians
    cov_nonzero = cov_by_mut.replace(0, np.nan).dropna()
    dropout_cut = float(np.quantile(cov_nonzero, float(bias_dropout_quantile))) if len(cov_nonzero) else 0.0
    flag_dropout = (cov_by_mut <= dropout_cut).rename("flag_dropout")

    # High-coverage threshold
    cov_q = float(np.quantile(df["coverage"], float(bias_min_highcov_quantile))) if len(df) else float(min_coverage)
    highcov_thr = max(cov_q, float(min_coverage))

    # Positive rate among high coverage rows
    df_highcov = df[df["coverage"] >= highcov_thr]
    if df_highcov.empty:
        pos_rate = pd.Series(0.0, index=cov_by_mut.index, name="af_pos_rate_highcov")
    else:
        df_highcov = df_highcov.copy()
        df_highcov["is_positive"] = (df_highcov["af"] > float(left_censor_af)).astype(float)
        pos_rate = (df_highcov.groupby("mutation")["is_positive"].mean()
                    .reindex(cov_by_mut.index).fillna(0.0)
                    .rename("af_pos_rate_highcov"))

    # Reference-bias flag
    flag_ref_bias = (pos_rate <= float(bias_af_pos_rate_max)).rename("flag_ref_bias")

    out = pd.concat([cov_by_mut, coverage_ratio, pos_rate, flag_dropout, flag_ref_bias], axis=1).reset_index()
    out["dropout_threshold"] = float(dropout_cut)
    out["highcov_threshold"] = float(highcov_thr)
    out["ref_bias_threshold"] = float(bias_af_pos_rate_max)
    out["global_median_coverage"] = float(global_cov)
    out["flag_dropout"] = out["flag_dropout"].astype(bool)
    out["flag_ref_bias"] = out["flag_ref_bias"].astype(bool)

    # Stable ordering: flagged up front
    out = out.sort_values(
        ["flag_dropout", "flag_ref_bias", "median_coverage", "af_pos_rate_highcov"],
        ascending=[False, False, True, True], kind="mergesort"
    ).reset_index(drop=True)

    return out

def bias_loci_figure(bias_df: pd.DataFrame) -> plt.Figure:
    """
    Bias‑loci diagnostics with a **shared legend baked UNDER the panels** (no overlap/clipping).

    Layout:
      ┌───────────────┬───────────────┐
      │   Panel A     │   Panel B     │   ← main plots
      └───────────────┴───────────────┘
      ┌───────────────────────────────┐
      │           LEGEND              │   ← dedicated legend row
      └───────────────────────────────┘

    Expects columns:
      Panel A: median_coverage, coverage_ratio_to_global, flag_dropout,
               dropout_threshold, global_median_coverage
      Panel B: median_coverage, af_pos_rate_highcov, flag_ref_bias,
               highcov_threshold
      Optional: ref_bias_threshold
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    set_matplotlib_style()

    # ------- Guards & required columns -------
    if bias_df is None or len(bias_df) == 0:
        fig, ax = plt.subplots(figsize=(12, 5.8))
        ax.axis("off")
        ax.text(0.5, 0.5, "No bias‑loci diagnostics available", ha="center", va="center")
        return fig

    df = bias_df.copy()
    req_a = ["median_coverage", "coverage_ratio_to_global", "flag_dropout",
             "dropout_threshold", "global_median_coverage"]
    req_b = ["median_coverage", "af_pos_rate_highcov", "flag_ref_bias",
             "highcov_threshold"]
    missing = [c for c in (req_a + req_b) if c not in df.columns]
    if missing:
        fig, ax = plt.subplots(figsize=(12, 5.8))
        ax.axis("off")
        ax.text(0.5, 0.5, "Missing required columns:\n" + ", ".join(missing),
                ha="center", va="center")
        return fig

    # ------- Helpers -------
    EPS = 1e-6

    def _log_limits(v: np.ndarray) -> tuple[float, float]:
        vv = np.asarray(v, dtype=float)
        vv = vv[np.isfinite(vv) & (vv > 0)]
        if vv.size == 0:
            return (1e-6, 1.0)
        if vv.size >= 10:
            lo = np.nanpercentile(vv, 1)
            hi = np.nanpercentile(vv, 99)
        else:
            lo, hi = float(vv.min()), float(vv.max())
        lo = max(EPS, lo * 0.90)
        hi = max(lo * 1.01, hi * 1.10)
        return (lo, hi)

    # Typed/clean views
    medcov      = df["median_coverage"].astype(float).to_numpy()
    covratio    = df["coverage_ratio_to_global"].astype(float).to_numpy()
    posrate     = df["af_pos_rate_highcov"].astype(float).to_numpy()
    flag_dropout = df["flag_dropout"].fillna(False).astype(bool).to_numpy()
    flag_refbias = df["flag_ref_bias"].fillna(False).astype(bool).to_numpy()

    # Thresholds (safe fallbacks)
    get1 = lambda col, default=np.nan: float(df[col].dropna().iloc[0]) \
        if col in df and df[col].notna().any() else float(default)
    dropout_thr = get1("dropout_threshold", 1.0)
    global_med  = get1("global_median_coverage", np.nan)
    highcov_thr = get1("highcov_threshold", 10.0)
    ref_bias_thr = get1("ref_bias_threshold", np.nan)

    # ------- Figure with dedicated bottom legend row -------
    fig = plt.figure(figsize=(12.4, 6.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.0, 0.26],   # bottom row reserved for the legend
        width_ratios=[1.0, 1.0],
        left=0.06, right=0.99, bottom=0.06, top=0.90,
        hspace=0.18, wspace=0.12
    )
    ax1   = fig.add_subplot(gs[0, 0])
    ax2   = fig.add_subplot(gs[0, 1])
    legax = fig.add_subplot(gs[1, :]); legax.axis("off")

    fig.suptitle("Bias Loci Diagnostics — Dropout vs Reference Bias", fontsize=14)

    # ------- Panel A: Amplicon dropout (log–log) -------
    ax1.set_xscale("log"); ax1.set_yscale("log")
    xA = np.clip(medcov, EPS, None)
    yA = np.clip(covratio, EPS, None)

    ax1.scatter(xA[~flag_dropout], yA[~flag_dropout],
                s=20, alpha=0.80, color="skyblue", label="Normal (dropout)", rasterized=True)
    ax1.scatter(xA[flag_dropout],  yA[flag_dropout],
                s=28, alpha=0.95, color="crimson",  label="Dropout‑flagged", rasterized=True)

    if np.isfinite(dropout_thr):
        ax1.axvline(max(EPS, dropout_thr), ls="--", lw=1.5, color="black",
                    label=f"Dropout cutoff = {dropout_thr:.0f}")
    if np.isfinite(global_med):
        ax1.axvline(max(EPS, global_med), ls=":", lw=1.2, color="gray",
                    label=f"Global median = {global_med:.0f}")
    ax1.axhline(1.0, ls=":", lw=1.0, color="0.55")

    xlo, xhi = _log_limits(xA); ylo, yhi = _log_limits(yA)
    ax1.set_xlim(xlo, xhi); ax1.set_ylim(ylo, yhi)
    ax1.set_title("A. Amplicon dropout detection")
    ax1.set_xlabel("Per‑mutation median coverage")
    ax1.set_ylabel("Coverage ratio to global median")
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    # ------- Panel B: Reference bias (x log, y linear) -------
    ax2.set_xscale("log")
    xB = xA
    yB = np.clip(posrate, 0.0, 1.0)

    ax2.scatter(xB[~flag_refbias], yB[~flag_refbias],
                s=20, alpha=0.80, color="seagreen",   label="Normal (ref‑bias)", rasterized=True)
    ax2.scatter(xB[flag_refbias],  yB[flag_refbias],
                s=28, alpha=0.95, color="darkorange", label="Ref‑bias flagged", rasterized=True)

    if np.isfinite(highcov_thr):
        ax2.axvline(max(EPS, highcov_thr), ls="--", lw=1.5, color="black",
                    label=f"High‑cov cutoff = {highcov_thr:.0f}")
    if np.isfinite(ref_bias_thr):
        ax2.axhline(ref_bias_thr, ls=":", lw=1.2, color="gray",
                    label=f"Ref‑bias threshold = {ref_bias_thr:.3f}")

    xb_lo, xb_hi = _log_limits(xB)
    ax2.set_xlim(xb_lo, xb_hi)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("B. Reference‑bias detection")
    ax2.set_xlabel("Per‑mutation median coverage (log scale)")
    ax2.set_ylabel("Positive AF rate (high coverage)")
    ax2.grid(True, which="both", ls=":", alpha=0.4)

    # ------- Shared legend centered UNDER the panels -------
    legend_handles = [
        Patch(facecolor="skyblue",   edgecolor="none", label="Normal (dropout)"),
        Patch(facecolor="crimson",   edgecolor="none", label="Dropout‑flagged"),
        Patch(facecolor="seagreen",  edgecolor="none", label="Normal (ref‑bias)"),
        Patch(facecolor="darkorange",edgecolor="none", label="Ref‑bias flagged"),
    ]
    legax.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="center",
        ncol=2,
        frameon=True, fancybox=True, framealpha=0.94,
        borderpad=0.6, handlelength=1.6, handletextpad=0.6, columnspacing=1.2
    )

    # No need for bbox tricks on save — legend is part of the figure canvas.
    return fig

# ---------------------------
# Figures
# ---------------------------
def coverage_panel_figure(df: pd.DataFrame, ridgeline_sites_max: int) -> plt.Figure:
    """
    Coverage panel with two views and a dedicated legend row UNDER the plots:

      (A) Global coverage distribution — normalized histogram + continuous KDE
      (B) Site-wise ridgeline densities — all scaled to a common peak height

    The legend lives on its own axes (bottom row), so exported PNGs never overlap/clip.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde

    set_matplotlib_style()

    # ---------- Guards ----------
    if df is None or df.empty or ("coverage" not in df.columns) or ("site_id" not in df.columns):
        fig, ax = plt.subplots(figsize=(11.6, 6.0))
        ax.axis("off")
        ax.text(0.5, 0.5, "No coverage data available", ha="center", va="center")
        return fig

    # ---------- Figure & layout: 2 plot rows + 1 legend row ----------
    fig = plt.figure(figsize=(11.8, 8.6))
    gs = fig.add_gridspec(
        nrows=3, ncols=1,
        height_ratios=[1.15, 1.0, 0.24],   # bottom row reserved for legend
        left=0.06, right=0.99, bottom=0.06, top=0.90, hspace=0.36
    )
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_ridge = fig.add_subplot(gs[1, 0])
    leg_ax  = fig.add_subplot(gs[2, 0]); leg_ax.axis("off")

    fig.suptitle("Coverage Panel — Global vs Site‑Level Distributions", fontsize=13)

    # ---------- (A) Global distribution ----------
    cov = df["coverage"].to_numpy(dtype=float)
    cov = cov[np.isfinite(cov) & (cov >= 0)]
    show_hist = False
    show_kde  = False

    if cov.size == 0:
        ax_hist.axis("off")
        ax_hist.text(0.5, 0.5, "No valid coverage values", ha="center", va="center")
    else:
        # Robust x‑limit
        x_max = float(np.nanpercentile(cov, 99.5))
        x_max = max(x_max, 1.0)

        # Bin count: Freedman–Diaconis (fallback to sqrt(n))
        q25, q75 = np.percentile(cov, [25, 75])
        iqr = max(q75 - q25, 0.0)
        if iqr > 0:
            binw = 2.0 * iqr * (cov.size ** (-1.0 / 3.0))
            bins = int(np.clip(np.ceil((x_max - 0.0) / max(binw, 1e-9)), 10, 60))
        else:
            bins = min(60, max(10, int(np.sqrt(cov.size))))

        # Histogram (density)
        ax_hist.hist(
            cov, bins=bins, range=(0.0, x_max), density=True,
            alpha=0.55, color="steelblue", edgecolor="none", label="Histogram (normalized)"
        )
        show_hist = True

        # KDE if there is variance
        if np.std(cov) > 0:
            xgrid = np.linspace(0.0, x_max, 512)
            try:
                kde = gaussian_kde(cov)
                ax_hist.plot(xgrid, kde(xgrid), lw=2.0, color="black", label="KDE (density)")
                show_kde = True
            except Exception:
                pass

        ax_hist.set_xlim(0.0, x_max)
        ax_hist.set_title("A. Global Coverage Distribution", fontsize=11)
        ax_hist.set_xlabel("Coverage")
        ax_hist.set_ylabel("Probability density")
        ax_hist.grid(True, ls=":", alpha=0.4)

    # ---------- (B) Ridgeline per site ----------
    sites = sorted(map(str, df["site_id"].dropna().astype(str).unique()))
    if len(sites) > ridgeline_sites_max:
        sites = sites[:ridgeline_sites_max]

    all_vals = df["coverage"].to_numpy(dtype=float)
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals >= 0)]
    if all_vals.size == 0:
        ax_ridge.axis("off")
        ax_ridge.text(0.5, 0.5, "No coverage values per site", ha="center", va="center")
    else:
        xgrid_r = np.linspace(0.0, float(np.nanpercentile(all_vals, 99.0)), 400)
        xgrid_r[-1] = max(xgrid_r[-1], 1.0)

        colors = categorical_palette(len(sites))
        site_densities = []
        global_ymax = 1e-12

        for site in sites:
            vals = df.loc[df["site_id"].astype(str) == site, "coverage"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size < 2 or np.std(vals) == 0:
                dens = np.zeros_like(xgrid_r)
            else:
                try:
                    kde = gaussian_kde(vals)
                    dens = kde(xgrid_r)
                except Exception:
                    dens = np.zeros_like(xgrid_r)
            site_densities.append((site, dens))
            if dens.size:
                global_ymax = max(global_ymax, float(np.max(dens)))

        y_spacing = 1.0 / max(1, len(sites))
        for i, (site, dens) in enumerate(site_densities):
            y0 = i * y_spacing
            color = colors[i % len(colors)]
            scaled = (dens / (global_ymax + 1e-12)) * y_spacing * 0.8
            ax_ridge.fill_between(xgrid_r, y0, y0 + scaled, color=color, alpha=0.5, lw=0)
            ax_ridge.plot(xgrid_r, y0 + scaled, color=color, lw=1.0)
            ax_ridge.text(xgrid_r[0], y0 + 0.02 * y_spacing, site,
                          va="bottom", ha="left", fontsize=8)

        ax_ridge.set_title("B. Coverage Ridgeline by Site", fontsize=11)
        ax_ridge.set_xlabel("Coverage")
        ax_ridge.set_yticks([])
        ax_ridge.set_xlim(left=0.0, right=float(np.nanpercentile(all_vals, 99.5)))
        ax_ridge.set_ylim(-0.02, 1.0)
        ax_ridge.grid(True, axis="x", ls=":", alpha=0.3)

    # ---------- Dedicated legend UNDER the plots ----------
    legend_handles, legend_labels = [], []
    if show_hist:
        legend_handles.append(Patch(facecolor="steelblue", alpha=0.55, edgecolor="none"))
        legend_labels.append("Histogram (normalized)")
    if show_kde:
        legend_handles.append(Line2D([0], [0], color="black", lw=2.0))
        legend_labels.append("KDE (density)")

    # Always render the legend axis so it reserves space in PNGs
    if legend_handles:
        leg_ax.legend(
            legend_handles, legend_labels,
            loc="center",
            ncol=len(legend_handles),
            frameon=True, fancybox=True, framealpha=0.94,
            borderpad=0.6, handlelength=1.8, handletextpad=0.6, columnspacing=1.2
        )
    else:
        leg_ax.text(0.5, 0.5, "", ha="center", va="center")

    return fig

def missingness_heatmap_figure(miss_heat: pd.DataFrame, heatmap_mutations_max: int) -> plt.Figure:
    """
    Missingness heatmap (sites × mutations), publication-ready, with a
    dedicated bottom row for the colorbar so it is always UNDER the plot
    (no overlap/clipping in saved PNGs).

    Guarantees
    ----------
    • True 0–1 scale (vmin=0, vmax=1) – no autoscaling artifacts
    • Deterministic ordering of rows/cols
    • Optional variance-based column down-selection (then stable alpha sort)
    • Tick thinning to keep labels readable on large matrices
    • Light cell gridlines for small/medium matrices
    • Horizontal colorbar in its own axes below the heatmap
    • Graceful handling of empty/degenerate inputs
    """
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    set_matplotlib_style()

    # ---------- Defensive copy & early exit ----------
    if miss_heat is None or (isinstance(miss_heat, pd.DataFrame) and miss_heat.empty):
        # Even for empty, return a simple one-panel figure (no extra row needed)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No missingness matrix to display", ha="center", va="center")
        return fig

    df = miss_heat.copy()

    # Ensure index/columns are strings (prevents mixed-type tick issues)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df = df.sort_index(axis=0)  # sites
    df = df.sort_index(axis=1)  # mutations

    # Coerce values to float within [0, 1]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if (df.values.min() < 0.0) or (df.values.max() > 1.0):
        df = df.clip(lower=0.0, upper=1.0)

    # ---------- Column down-selection by variance (then stable alpha sort) ----------
    if df.shape[1] > heatmap_mutations_max:
        variances = df.var(axis=0)
        means = df.mean(axis=0)
        rank_df = (
            pd.DataFrame({"var": variances, "mean": means})
              .sort_values(["var", "mean"], ascending=[False, False])
              .head(heatmap_mutations_max)
        )
        keep_cols = sorted(rank_df.index.tolist())
        df = df.loc[:, keep_cols]

    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])

    # ---------- Figure sizing ----------
    # Scale sensibly with matrix size, with upper bounds; reserve a bottom row
    # solely for the colorbar so it never overlaps tick labels in exported images.
    fig_height = min(18.0, max(4.4, 0.28 * max(n_rows, 1) + 1.8))
    fig_width  = min(22.0, max(8.6, 0.16 * max(n_cols, 1) + 6.2))
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Two rows: heatmap (top), colorbar (bottom).
    # Bottom row height is fixed proportion so even long x‑tick labels won’t collide.
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.0, 0.20],   # dedicate ~20% height for cb + x‑tick labels
        left=0.06, right=0.98, top=0.94, bottom=0.08, hspace=0.18
    )
    ax  = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 0])   # dedicated colorbar axes underneath

    # Colormap with graceful fallback
    try:
        cmap = continuous_cmap("cet_fire")  # from utils.plotting if available
    except Exception:
        cmap = plt.cm.get_cmap("magma")

    # ---------- Heatmap ----------
    im = ax.imshow(
        df.values,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
    )

    # Titles/labels
    ax.set_title("Missingness heatmap (sites × mutations)")
    ax.set_xlabel("Mutation")
    ax.set_ylabel("Site")

    # ---------- Tick thinning ----------
    def _thin_ticks(n: int, target: int = 36) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=int)
        if n <= target:
            return np.arange(n, dtype=int)
        step = int(math.ceil(n / target))
        return np.arange(0, n, step, dtype=int)

    xticks = _thin_ticks(n_cols, target=36)
    yticks = _thin_ticks(n_rows, target=36)

    ax.set_xticks(xticks)
    ax.set_xticklabels([df.columns[i] for i in xticks],
                       rotation=90, fontsize=7 if n_cols <= 120 else 6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([df.index[i] for i in yticks],
                       fontsize=8 if n_rows <= 120 else 7)

    # Short ticks; lighter frame
    ax.tick_params(length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_alpha(0.6)

    # Optional gridlines for smaller matrices
    if n_rows * n_cols <= 60 * 60:
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)

    # ---------- Colorbar UNDER the heatmap (dedicated axes) ----------
    # Placing it in cax guarantees it sits below the plot without overlapping.
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Missingness rate")
    cbar_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_xticklabels([f"{t:.2f}" for t in cbar_ticks])

    # No tight_layout: geometry is fully controlled by GridSpec (prevents cb overlap).
    return fig

def alt_ref_scatter_figure(
    df: pd.DataFrame,
    left_censor_af: float,
    label_suffix: Optional[str] = None
) -> plt.Figure:
    """
    Alt (y) vs Ref (x) with a dedicated legend row UNDER the plot (no overlap/clipping in PNGs).
    Keeps the math/overlays identical to your version; only the layout is changed.
    """
    import math
    from collections import OrderedDict

    set_matplotlib_style()

    # ---- Figure: main plot (top) + legend row (bottom) ----
    fig = plt.figure(figsize=(10.8, 6.8))
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.0, 0.24],  # bottom row reserved for legend
        left=0.08, right=0.99, bottom=0.06, top=0.90, hspace=0.02
    )
    ax    = fig.add_subplot(gs[0, 0])
    legax = fig.add_subplot(gs[1, 0]); legax.axis("off")

    # ---------- Prepare counts (defensive coercion) ----------
    tmp = df.copy()
    for c in ("count", "coverage"):
        tmp[c] = pd.to_numeric(tmp.get(c, 0), errors="coerce").fillna(0.0).astype(float)
        tmp.loc[tmp[c] < 0, c] = 0.0

    tmp["ref_count"] = np.maximum(tmp["coverage"] - tmp["count"], 0.0)
    alt = tmp["count"].to_numpy(dtype=float)
    ref = tmp["ref_count"].to_numpy(dtype=float)
    cov = tmp["coverage"].to_numpy(dtype=float)

    valid = np.isfinite(alt) & np.isfinite(ref) & np.isfinite(cov) & (alt >= 0) & (ref >= 0) & (cov >= 0)
    if not np.any(valid):
        ax.text(0.5, 0.5, "No valid counts", ha="center", va="center")
        ax.set_xlabel("Alt count"); ax.set_ylabel("Ref count")
        fig.suptitle("Alt vs Ref counts" + ("" if not label_suffix else f" – {label_suffix}"), fontsize=13)
        return fig

    alt, ref, cov = alt[valid], ref[valid], cov[valid]
    cov_pos = cov[cov > 0]

    # ---------- Robust axis scale ----------
    m = float(np.percentile(cov_pos, 99.5)) if cov_pos.size else 1.0
    lim = max(1.0, m) * 1.05

    # ---------- Scatter ----------
    ax.scatter(alt, ref, s=10, alpha=0.45, rasterized=True, label="Rows")

    # ---------- AF isoclines: y = ((1-f)/f) * x ----------
    x = np.linspace(0, lim, 512)
    base_afs = [0.5, 0.1]
    if left_censor_af and 0.0 < float(left_censor_af) < 1.0:
        base_afs.append(float(left_censor_af))
    iso_afs = sorted(set(f for f in base_afs if 0.0 < f < 1.0), reverse=True)

    styles = ["--", ":", "-.", (0, (3, 1))]
    for i, f in enumerate(iso_afs):
        slope = (1.0 / f) - 1.0  # = (1-f)/f
        ax.plot(x, slope * x, styles[i % len(styles)], lw=1.3, color="black", label=f"{f*100:.1f}% AF")

    # ---------- Iso-coverage lines: ref = c - alt for c in quantiles ----------
    if cov_pos.size:
        qs = np.quantile(cov_pos, [0.25, 0.5, 0.75, 0.90]).astype(float)
        qs = np.unique(np.clip(np.round(qs, 0), 1.0, None))  # dedup rounded
        for c, style in zip(qs, ["-", "--", ":", "-."]):
            xs = np.linspace(0, min(lim, c), 2)
            ax.plot(xs, c - xs, style, lw=1.0, color="grey", alpha=0.95, label=f"Coverage = {c:.0f}")

    # ---------- Axes, grid, cosmetics ----------
    suffix = "" if not label_suffix else f" – {label_suffix}"
    ax.set_title(f"Alt vs Ref counts{suffix}")
    ax.set_xlabel("Alt count"); ax.set_ylabel("Ref count")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", alpha=0.35)
    ax.margins(x=0.02, y=0.02)
    for spine in ax.spines.values():
        spine.set_alpha(0.6)

    # ---------- Stats block (upper-right, out of the way) ----------
    K = float(np.sum(alt)); N = float(np.sum(cov)); p_hat = K / max(N, 1.0)
    z = 1.959963984540054
    if N > 0.0:
        denom = 1.0 + (z ** 2) / N
        center = (p_hat + (z ** 2) / (2.0 * N)) / denom
        half = (z / denom) * math.sqrt((p_hat * (1 - p_hat)) / max(N, 1.0) + (z ** 2) / (4.0 * (N ** 2)))
        ci_lo, ci_hi = max(0.0, center - half), min(1.0, center + half)
    else:
        ci_lo = ci_hi = float("nan")

    try:
        pr = pearsonr(alt, ref); pear_r = float(getattr(pr, "statistic", pr[0])); pear_p = float(getattr(pr, "pvalue", pr[1]))
    except Exception:
        pear_r = pear_p = float("nan")
    try:
        sr = spearmanr(alt, ref); spear_r = float(getattr(sr, "statistic", sr[0])); spear_p = float(getattr(sr, "pvalue", sr[1]))
    except Exception:
        spear_r = spear_p = float("nan")

    hi_thr = float(np.quantile(cov, 0.70)) if cov.size else 0.0
    mask_hi = cov >= hi_thr
    if np.any(mask_hi):
        af_hi = np.divide(alt[mask_hi], np.maximum(cov[mask_hi], 1.0))
        pos_rate_hi = float(np.mean(af_hi > float(left_censor_af)))
        zero_alt_hi = float(np.mean(alt[mask_hi] == 0.0))
    else:
        pos_rate_hi = float("nan"); zero_alt_hi = float("nan")

    var_y = float(np.var(alt, ddof=1)) if alt.size >= 2 else float("nan")
    n_bar = float(np.mean(cov)) if cov.size else float("nan")
    denom_var = n_bar * p_hat * (1 - p_hat) if (n_bar > 0 and 0 < p_hat < 1) else float("nan")
    if np.isfinite(var_y) and np.isfinite(denom_var) and denom_var > 0:
        term = (var_y / max(denom_var, 1e-12)) - 1.0
        kappa = (n_bar - 1.0) / max(term, 1e-12) - 1.0
        phi_mom = float(max(0.0, kappa))
    else:
        phi_mom = float("nan")

    if cov_pos.size:
        q25, q50, q75, q90 = (float(q) for q in np.quantile(cov_pos, [0.25, 0.5, 0.75, 0.90]))
    else:
        q25 = q50 = q75 = q90 = float("nan")

    stats_text = (
        f"Global AF p̂ = {p_hat:.4f}  (Wilson 95% CI: {ci_lo:.4f}–{ci_hi:.4f})\n"
        f"Pearson r = {pear_r:+.3f}  (p={pear_p:.2g})   Spearman ρ = {spear_r:+.3f}  (p={spear_p:.2g})\n"
        f"High-cov q70 = {hi_thr:.0f}  |  P(AF>{left_censor_af:.3f} | cov≥thr) = {pos_rate_hi:.3f}   zeros = {zero_alt_hi:.3f}\n"
        f"Overdispersion φ_mom ≈ {phi_mom:.3f}\n"
        f"Coverage: p25={q25:.0f}, p50={q50:.0f}, p75={q75:.0f}, p90={q90:.0f}"
    )
    ax.text(
        0.98, 0.98, stats_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.85", alpha=0.95)
    )

    # ---------- Legend UNDER the plot (boxed, deduped) ----------
    handles, labels = ax.get_legend_handles_labels()
    dedup = OrderedDict()
    for h, l in zip(handles, labels):
        if l not in dedup:
            dedup[l] = h

    if dedup:
        ncols = min(6, max(2, int(math.ceil(len(dedup) / 2))))
        legax.legend(
            list(dedup.values()), list(dedup.keys()),
            loc="center", ncol=ncols,
            frameon=True, fancybox=True, framealpha=0.92,
            borderpad=0.6, handletextpad=0.6, columnspacing=1.0
        )
    else:
        legax.text(0.5, 0.5, "", ha="center", va="center")

    fig.suptitle("Alt vs Ref counts" + suffix, fontsize=13)
    return fig

def coverage_by_mutation_figure(df: pd.DataFrame) -> plt.Figure:
    """
    Coverage per mutation with a dedicated legend row UNDER the plot (no overlap/clipping).
    If seaborn is available, the CV colorbar is docked in a narrow slot to the right.
    """
    import math
    import scipy.stats as st
    from collections import OrderedDict

    set_matplotlib_style()

    # ---------- Defensive copy & coercion ----------
    d = df.copy()
    if d.empty or "mutation" not in d.columns or "coverage" not in d.columns:
        fig, ax = plt.subplots(figsize=(10.8, 6.2))
        ax.text(0.5, 0.5, "No data for coverage by mutation", ha="center", va="center")
        ax.set_axis_off()
        return fig

    d["mutation"] = d["mutation"].astype(str)
    d["coverage"] = pd.to_numeric(d["coverage"], errors="coerce").fillna(0.0).astype(float)
    d.loc[d["coverage"] < 0, "coverage"] = 0.0

    # ---------- Per-mutation statistics ----------
    grp = d.groupby("mutation", sort=True)["coverage"]
    stats = grp.agg(count="count", mean="mean", median="median", var="var").rename(columns={"var": "variance"})
    stats["cv"]  = np.sqrt(stats["variance"]).divide(stats["mean"].replace(0, np.nan))
    stats["mad"] = grp.apply(lambda x: float(np.median(np.abs(x - np.median(x)))))
    q = grp.quantile([0.05, 0.95]).unstack() if len(grp) else pd.DataFrame(columns=[0.05, 0.95])
    stats["q05"] = q.get(0.05, np.nan)
    stats["q95"] = q.get(0.95, np.nan)

    stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    stats = stats.sort_values("median", ascending=True)
    order = stats.index.tolist()
    if not order:
        fig, ax = plt.subplots(figsize=(10.8, 6.2))
        ax.text(0.5, 0.5, "No mutations after filtering", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # ---------- Global summary ----------
    cov_all = d["coverage"].to_numpy(dtype=float)
    cov_all = cov_all[np.isfinite(cov_all) & (cov_all >= 0)]
    if cov_all.size == 0:
        cov_all = np.array([0.0])

    global_median = float(np.median(cov_all))
    global_mad    = float(np.median(np.abs(cov_all - global_median)))
    global_var    = float(np.var(cov_all, ddof=1)) if cov_all.size >= 2 else 0.0
    global_skew   = float(st.skew(cov_all, nan_policy="omit")) if np.nanstd(cov_all) > 0 else 0.0

    q05, q95 = np.nanpercentile(cov_all, [5, 95])
    log_mode = (q95 / max(q05, 1.0)) >= 50.0
    y_min = 1.0 if log_mode else 0.0
    y_max = float(np.nanpercentile(cov_all, 99.5)) * 1.05
    if not np.isfinite(y_max) or y_max <= y_min:
        y_max = y_min + 1.0  # safe fallback

    # ---------- Figure: plot (left) + cbar slot (right, top row), legend row (bottom) ----------
    n = len(order)
    base_w = max(8.0, 0.18 * n + 6.0)
    fig = plt.figure(figsize=(base_w + 3.2, 6.8))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.0, 0.05], height_ratios=[1.0, 0.26],
        left=0.05, right=0.99, bottom=0.06, top=0.90, wspace=0.18, hspace=0.02
    )
    ax    = fig.add_subplot(gs[0, 0])
    cax   = fig.add_subplot(gs[0, 1])   # used only when seaborn is available
    legax = fig.add_subplot(gs[1, :]); legax.axis("off")

    # ---------- Plot: violin (if seaborn) else boxplot ----------
    if HAS_SEABORN:
        sns.violinplot(
            x="mutation", y="coverage", data=d, order=order, ax=ax,
            inner=None, cut=0, bw="scott", color="lightsteelblue"
        )
        xs  = np.arange(n)
        med = stats["median"].to_numpy()
        cv  = stats["cv"].fillna(0.0).clip(0, 1.5).to_numpy()
        sc = ax.scatter(
            xs, med,
            s=40 + 110 * np.clip(cv, 0, 1),
            c=cv, cmap="viridis",
            edgecolor="black", lw=0.3, zorder=3,
            label="Median (size ∝ CV)"
        )
        # Dock the colorbar in the narrow slot (never overlaps legend)
        cb = fig.colorbar(sc, cax=cax, orientation="vertical")
        cb.set_label("Coefficient of variation (CV)")
        ax.set_xticks(xs)
        ax.set_xticklabels(order)
    else:
        data = [d.loc[d["mutation"] == m, "coverage"] for m in order]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], categorical_palette(n)):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.set_xticks(np.arange(1, n + 1))
        ax.set_xticklabels(order)
        # Add a median marker so we still get a legend entry
        med = stats["median"].to_numpy()
        xs = np.arange(1, n + 1)
        ax.scatter(xs, med, s=52, facecolor="white", edgecolor="black", lw=0.6, zorder=3, label="Median")

    # ---------- Reference line & axes ----------
    ax.axhline(global_median, color="black", ls="--", lw=1.2, label=f"Global median = {global_median:.0f}")
    if log_mode:
        ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Mutation")
    ax.set_ylabel("Coverage (log scale)" if log_mode else "Coverage")
    ax.set_title("Coverage distribution per mutation")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    # ---------- Tick label rotation & thinning ----------
    rot = 90 if n > 20 else 45
    xticklabels = ax.get_xticklabels()
    if n > 60:
        step = int(math.ceil(n / 60))
        for i, lab in enumerate(xticklabels):
            if i % step != 0:
                lab.set_visible(False)
    for lab in xticklabels:
        lab.set_rotation(rot)
        lab.set_horizontalalignment("right")

    # ---------- Global stats annotation ----------
    summary = (
        f"Global statistics:\n"
        f"Median ± MAD = {global_median:.0f} ± {global_mad:.0f}\n"
        f"Variance = {global_var:.1f}\n"
        f"Skewness = {global_skew:+.2f}\n"
        f"n_mutations = {n}"
    )
    ax.text(
        0.995, 0.02, summary,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="0.85", alpha=0.95)
    )

    # ---------- Legend UNDER the plot (boxed, deduped) ----------
    handles, labels = ax.get_legend_handles_labels()
    dedup = OrderedDict()
    for h, l in zip(handles, labels):
        if l not in dedup:
            dedup[l] = h

    if dedup:
        ncols = min(6, max(2, int(math.ceil(len(dedup) / 2))))
        legax.legend(
            list(dedup.values()), list(dedup.keys()),
            loc="center", ncol=ncols,
            frameon=True, fancybox=True, framealpha=0.92,
            borderpad=0.6, handletextpad=0.6, columnspacing=1.0
        )
    else:
        # Keep bottom row height even without entries
        legax.text(0.5, 0.5, "", ha="center", va="center")

    return fig

def coverage_ecdf_by_site_figure(df: pd.DataFrame) -> plt.Figure:
    """
    Coverage ECDF by site with a dedicated legend row UNDER the plot
    (baked into the PNG, no overlap/clipping). Math and visuals match your
    current version; only the layout is changed to guarantee legend placement.
    """
    import math
    from collections import OrderedDict

    set_matplotlib_style()

    # ---------- Defensive copy ----------
    d = df.copy()
    if d.empty or "site_id" not in d.columns or "coverage" not in d.columns:
        fig, ax = plt.subplots(figsize=(10.8, 6.2))
        ax.text(0.5, 0.5, "No data for Coverage ECDF by Site", ha="center", va="center")
        ax.set_axis_off()
        return fig

    d["site_id"] = d["site_id"].astype(str)
    d["coverage"] = pd.to_numeric(d["coverage"], errors="coerce").fillna(0.0).astype(float)
    d.loc[d["coverage"] < 0, "coverage"] = 0.0

    sites = sorted(d["site_id"].unique().tolist())
    n_sites = len(sites)
    colors = categorical_palette(n_sites)

    # ---------- Figure: main plot (top) + legend row (bottom) ----------
    fig = plt.figure(figsize=(10.8, 6.8))
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.0, 0.26],   # bottom row reserved for legend
        left=0.08, right=0.99, bottom=0.06, top=0.90, hspace=0.02
    )
    ax    = fig.add_subplot(gs[0, 0])
    legax = fig.add_subplot(gs[1, 0]); legax.axis("off")

    # ---------- Log-scale decision with guard ----------
    cov_all = d["coverage"].to_numpy(dtype=float)
    cov_all = cov_all[np.isfinite(cov_all) & (cov_all >= 0)]
    if cov_all.size == 0:
        cov_all = np.array([0.0])

    q05, q95 = np.percentile(cov_all, [5, 95])
    use_log = (q95 / max(q05, 1.0)) >= 50.0

    pos_all = cov_all[cov_all > 0]
    if use_log and pos_all.size == 0:
        use_log = False

    if use_log:
        ax.set_xscale("log")
        x_min = max(1e-3, float(np.percentile(pos_all, 0.1)) * 0.5) if pos_all.size else 1.0
        x_max = float(np.percentile(pos_all, 99.7)) * 1.05 if pos_all.size else 1.0
        if not np.isfinite(x_max) or x_max <= x_min:
            x_max = x_min * 10.0
    else:
        x_min = 0.0
        x_max = float(np.percentile(cov_all, 99.7)) * 1.05
        if not np.isfinite(x_max) or x_max <= 0:
            x_max = 1.0

    # ---------- ECDF per site ----------
    handles = []
    labels = []
    lw = 1.6 if n_sites <= 30 else (1.2 if n_sites <= 60 else 1.0)

    stats_rows = []
    draw_median_lines = n_sites <= 40  # avoid clutter if many sites

    for i, site in enumerate(sites):
        vals = d.loc[d["site_id"] == site, "coverage"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals >= 0)]
        n = vals.size
        if n == 0:
            continue

        vals = np.sort(vals)
        # ECDF y
        y = np.arange(1, n + 1, dtype=float) / n

        # Log plotting guard: clip zeros up to x_min
        vals_plot = np.clip(vals, x_min if use_log else 0.0, None)

        # Plot ECDF
        (line,) = ax.step(vals_plot, y, where="post", lw=lw, color=colors[i], alpha=0.95, label=site)
        handles.append(line); labels.append(site)

        # Statistics
        mean = float(np.mean(vals))
        var  = float(np.var(vals, ddof=1)) if n >= 2 else float("nan")
        med  = float(np.median(vals))
        q10, q90 = np.percentile(vals, [10, 90]) if n else (float("nan"), float("nan"))
        if vals.sum() > 0:
            idx = np.arange(1, n + 1, dtype=float)
            gini = 2 * np.sum(idx * vals) / (n * np.sum(vals)) - (n + 1) / n
        else:
            gini = float("nan")
        stats_rows.append((site, n, mean, med, var, q10, q90, gini))

        # Median line
        if draw_median_lines and np.isfinite(med) and (med > 0 or not use_log):
            ax.axvline(max(med, x_min if use_log else med),
                       ls="--", lw=0.8, color=colors[i], alpha=0.55)

    # ---------- Summary annotation (top-right of the plot) ----------
    stats_df = pd.DataFrame(stats_rows, columns=["site", "n", "mean", "median", "var", "q10", "q90", "gini"])
    if not stats_df.empty:
        summary = (
            "Global ECDF statistics (median across sites):\n"
            f"Mean = {stats_df['mean'].median():.1f}\n"
            f"Median = {stats_df['median'].median():.1f}\n"
            f"Variance = {stats_df['var'].median():.1f}\n"
            f"q10 = {stats_df['q10'].median():.1f}, q90 = {stats_df['q90'].median():.1f}\n"
            f"Gini = {stats_df['gini'].median():.3f}"
        )
        ax.text(
            0.98, 0.98, summary,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                      edgecolor="0.85", alpha=0.95)
        )

    # ---------- Axes formatting ----------
    ax.set_title("Coverage ECDF by Site")
    ax.set_xlabel("Coverage" + (" (log scale)" if use_log else ""))
    ax.set_ylabel("Empirical CDF (F(x))")
    ax.set_xlim(x_min if use_log else 0.0, x_max)
    ax.set_ylim(0, 1.02)
    ax.grid(True, ls=":", alpha=0.4)

    # ---------- Legend UNDER the plot (boxed, dedup + size cap) ----------
    if handles:
        dedup = OrderedDict()
        for h, l in zip(handles, labels):
            if l not in dedup:
                dedup[l] = h

        max_items = 30
        shown_items  = list(dedup.items())[:max_items]
        shown_labels = [l for l, _ in shown_items]
        shown_handles = [h for _, h in shown_items]
        if len(dedup) > max_items:
            from matplotlib.lines import Line2D
            shown_labels.append(f"+{len(dedup) - max_items} more")
            shown_handles.append(Line2D([], [], color="none"))

        ncols = min(6, max(2, int(math.ceil(len(shown_labels) / 2))))
        legax.legend(
            shown_handles, shown_labels,
            loc="center",
            ncol=ncols,
            frameon=True, fancybox=True, framealpha=0.92,
            borderpad=0.6, handletextpad=0.6, columnspacing=1.0
        )
    else:
        legax.text(0.5, 0.5, "", ha="center", va="center")

    return fig

def prevalence_entropy_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-mutation prevalence and temporal dispersion (entropy), robust and schema-stable.

    Returns
    -------
    prevalence : DataFrame
        Columns:
          ['mutation', 'prevalence_unweighted', 'prevalence_weighted',
           'n_rows', 'total_count', 'total_coverage',
           'n_dates_total', 'n_dates_nonzero',
           'mean_coverage', 'median_coverage',
           'first_date', 'last_date']
    entropy : DataFrame
        Columns:
          ['mutation', 'temporal_entropy_normalized', 'temporal_effective_num_dates',
           'temporal_gini', 'temporal_peak_date', 'temporal_peak_fraction',
           'temporal_n_dates', 'temporal_n_nonzero_dates']
    """
    # ---------- Defensive copy & required columns ----------
    d = df.copy()
    req = {"mutation", "count", "coverage"}
    if not req.issubset(d.columns):
        prevalence_cols = [
            "mutation", "prevalence_unweighted", "prevalence_weighted",
            "n_rows", "total_count", "total_coverage",
            "n_dates_total", "n_dates_nonzero",
            "mean_coverage", "median_coverage",
            "first_date", "last_date",
        ]
        entropy_cols = [
            "mutation", "temporal_entropy_normalized", "temporal_effective_num_dates",
            "temporal_gini", "temporal_peak_date", "temporal_peak_fraction",
            "temporal_n_dates", "temporal_n_nonzero_dates",
        ]
        return (pd.DataFrame(columns=prevalence_cols),
                pd.DataFrame(columns=entropy_cols))

    # Types & cleaning
    d["mutation"] = d["mutation"].astype(str)
    d["count"]    = pd.to_numeric(d["count"], errors="coerce").fillna(0).astype(float)
    d["coverage"] = pd.to_numeric(d["coverage"], errors="coerce").fillna(0).astype(float)
    d.loc[d["count"] < 0, "count"]       = 0.0
    d.loc[d["coverage"] < 0, "coverage"] = 0.0

    # If af_obs missing, compute it (KEEP-ALL semantics); clamp to [0,1]
    if "af_obs" not in d.columns:
        d["af_obs"] = np.where(d["coverage"] > 0, d["count"] / d["coverage"], 0.0)
    else:
        d["af_obs"] = pd.to_numeric(d["af_obs"], errors="coerce").fillna(0.0).astype(float)
    d["af_obs"] = d["af_obs"].clip(0.0, 1.0)

    # ---------- Prevalence ----------
    # Unweighted prevalence: fraction of rows with AF>0
    prev_unw = (d["af_obs"] > 0).groupby(d["mutation"]).mean().astype(float)
    prev_unw = prev_unw.rename("prevalence_unweighted")

    # Weighted prevalence: sum(count)/sum(coverage), clipped to [0,1]
    sums = d.groupby("mutation", observed=True)[["count", "coverage"]].sum(min_count=1)
    wprev = (sums["count"] / sums["coverage"].replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)
    wprev = wprev.astype(float).rename("prevalence_weighted")

    # Extras
    n_rows     = d.groupby("mutation", observed=True).size().rename("n_rows").astype(int)
    total_count = sums["count"].rename("total_count").astype(float)
    total_cov   = sums["coverage"].rename("total_coverage").astype(float)
    mean_cov    = d.groupby("mutation", observed=True)["coverage"].mean().rename("mean_coverage").astype(float)
    median_cov  = d.groupby("mutation", observed=True)["coverage"].median().rename("median_coverage").astype(float)

    # Dates (optional)
    has_date = "date" in d.columns
    if has_date:
        dd = d.copy()
        dd["date"] = pd.to_datetime(dd["date"], errors="coerce").dt.normalize()
        first_date = dd.groupby("mutation", observed=True)["date"].min().rename("first_date")
        last_date  = dd.groupby("mutation", observed=True)["date"].max().rename("last_date")
    else:
        first_date = pd.Series(dtype="datetime64[ns]", name="first_date")
        last_date  = pd.Series(dtype="datetime64[ns]", name="last_date")

    # Assemble prevalence table
    prevalence = (
        pd.concat(
            [prev_unw, wprev, n_rows, total_count, total_cov,
             mean_cov, median_cov, first_date, last_date],
            axis=1
        )
        .reset_index()
        .rename(columns={"index": "mutation"})
        .sort_values("mutation", kind="mergesort")
        .reset_index(drop=True)
    )

    # Initialize date-count columns; fill later if dates exist
    prevalence["n_dates_total"]   = 0
    prevalence["n_dates_nonzero"] = 0

    # ---------- Temporal entropy & related metrics ----------
    ent_rows = []

    if has_date:
        md = (
            d.assign(date=pd.to_datetime(d["date"], errors="coerce").dt.normalize())
             .dropna(subset=["date"])
             .groupby(["mutation", "date"], observed=True)[["count", "coverage"]]
             .sum(min_count=1)
             .reset_index()
        )
        md["af_date"] = np.where(md["coverage"] > 0, md["count"] / md["coverage"], 0.0).clip(0.0, 1.0)

        # n_dates_total / n_dates_nonzero for prevalence
        n_dates_total = md.groupby("mutation", observed=True)["date"].nunique().rename("n_dates_total").astype(int)
        n_dates_nonzero = (
            md.assign(_nz=md["af_date"] > 0)
              .groupby("mutation", observed=True)["_nz"].sum()
              .rename("n_dates_nonzero").astype(int)
        )
        prevalence = (
            prevalence.set_index("mutation")
                      .combine_first(pd.concat([n_dates_total, n_dates_nonzero], axis=1))
                      .reset_index()
        )
        for col in ("n_dates_total", "n_dates_nonzero"):
            if col in prevalence.columns:
                prevalence[col] = pd.to_numeric(prevalence[col], errors="coerce").fillna(0).astype(int)

        # Entropy table per mutation
        for mut, g in md.groupby("mutation", sort=False, observed=True):
            af_series = g["af_date"].to_numpy(dtype=float)
            dates     = g["date"].to_numpy()
            n_dates   = int(af_series.size)

            if n_dates == 0:
                ent_rows.append({
                    "mutation": mut,
                    "temporal_entropy_normalized": 0.0,
                    "temporal_effective_num_dates": 0.0,
                    "temporal_gini": np.nan,
                    "temporal_peak_date": pd.NaT,
                    "temporal_peak_fraction": 0.0,
                    "temporal_n_dates": 0,
                    "temporal_n_nonzero_dates": 0,
                })
                continue

            total_mass = float(af_series.sum())
            if total_mass <= 0:
                ent_rows.append({
                    "mutation": mut,
                    "temporal_entropy_normalized": 0.0,
                    "temporal_effective_num_dates": 0.0,
                    "temporal_gini": np.nan,
                    "temporal_peak_date": pd.NaT,
                    "temporal_peak_fraction": 0.0,
                    "temporal_n_dates": n_dates,
                    "temporal_n_nonzero_dates": 0,
                })
                continue

            p = (af_series / total_mass).astype(float)
            nz_mask   = p > 0
            n_nonzero = int(nz_mask.sum())

            H     = float(-np.sum(p[nz_mask] * np.log(p[nz_mask] + 1e-12)))
            H_max = float(np.log(max(n_nonzero, 1)))
            H_norm = float(H / H_max) if H_max > 0 else 0.0
            n_eff  = float(np.exp(H))

            p_sorted = np.sort(p)
            idx = np.arange(1, len(p_sorted) + 1, dtype=float)
            gini = float(2.0 * np.sum(idx * p_sorted) / len(p_sorted) - (len(p_sorted) + 1.0) / len(p_sorted))

            peak_idx  = int(np.argmax(p))
            peak_date = pd.to_datetime(dates[peak_idx]) if dates.size else pd.NaT
            peak_frac = float(p[peak_idx])

            ent_rows.append({
                "mutation": mut,
                "temporal_entropy_normalized": H_norm,
                "temporal_effective_num_dates": n_eff,
                "temporal_gini": gini,
                "temporal_peak_date": peak_date,
                "temporal_peak_fraction": peak_frac,
                "temporal_n_dates": n_dates,
                "temporal_n_nonzero_dates": n_nonzero,
            })
    else:
        # No dates → consistent zeros/NaNs
        for mut in sorted(d["mutation"].unique()):
            ent_rows.append({
                "mutation": mut,
                "temporal_entropy_normalized": 0.0,
                "temporal_effective_num_dates": 0.0,
                "temporal_gini": np.nan,
                "temporal_peak_date": pd.NaT,
                "temporal_peak_fraction": 0.0,
                "temporal_n_dates": 0,
                "temporal_n_nonzero_dates": 0,
            })

    entropy = (
        pd.DataFrame(ent_rows)
          .sort_values("mutation", kind="mergesort")
          .reset_index(drop=True)
    )

    # Ensure consistent dtypes & clipping
    float_cols_prev = ["prevalence_unweighted", "prevalence_weighted",
                       "total_count", "total_coverage", "mean_coverage", "median_coverage"]
    for c in float_cols_prev:
        if c in prevalence.columns:
            prevalence[c] = pd.to_numeric(prevalence[c], errors="coerce").astype(float).fillna(0.0)

    int_cols_prev = ["n_rows", "n_dates_total", "n_dates_nonzero"]
    for c in int_cols_prev:
        if c in prevalence.columns:
            prevalence[c] = pd.to_numeric(prevalence[c], errors="coerce").fillna(0).astype(int)

    if "prevalence_weighted" in prevalence.columns:
        prevalence["prevalence_weighted"] = prevalence["prevalence_weighted"].clip(0.0, 1.0)

    return prevalence, entropy
def mutation_variance_and_missingness(
    df: pd.DataFrame,
    min_coverage: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-mutation AF variance and per-mutation missingness across the full site×date grid.

    Returns
    -------
    var_tbl : DataFrame
        ['mutation', 'var_af_obs', 'mean_af_obs', 'median_af_obs', 'iqr_af_obs',
         'wmean_af_obs', 'wvar_af_obs', 'n_obs', 'sum_coverage']
         (sorted by var_af_obs desc)
    miss_tbl : DataFrame
        ['mutation', 'missingness_rate', 'present_rate', 'present_count',
         'grid_size', 'n_sites', 'n_dates']
         (sorted by missingness_rate desc)
    """
    # -------- Defensive copy + schema checks --------
    d = df.copy()
    need = {"mutation", "coverage"}
    if not need.issubset(d.columns):
        var_tbl = pd.DataFrame(columns=[
            "mutation", "var_af_obs", "mean_af_obs", "median_af_obs", "iqr_af_obs",
            "wmean_af_obs", "wvar_af_obs", "n_obs", "sum_coverage"
        ])
        miss_tbl = pd.DataFrame(columns=[
            "mutation", "missingness_rate", "present_rate",
            "present_count", "grid_size", "n_sites", "n_dates"
        ])
        return var_tbl, miss_tbl

    # -------- Normalize types --------
    d["mutation"] = d["mutation"].astype(str)
    d["coverage"] = pd.to_numeric(d["coverage"], errors="coerce").fillna(0).astype(float)
    d.loc[d["coverage"] < 0, "coverage"] = 0.0
    if "count" in d.columns:
        d["count"] = pd.to_numeric(d["count"], errors="coerce").fillna(0).astype(float)
        d.loc[d["count"] < 0, "count"] = 0.0
    # Ensure af_obs exists
    if "af_obs" not in d.columns:
        d["af_obs"] = np.where(d.get("count", 0.0) > 0,
                               np.where(d["coverage"] > 0, d.get("count", 0.0) / d["coverage"], 0.0),
                               0.0)
    d["af_obs"] = pd.to_numeric(d["af_obs"], errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)

    # -------- VARIANCE & DESCRIPTIVE STATS (per mutation) --------
    if d.empty:
        var_tbl = pd.DataFrame(columns=[
            "mutation", "var_af_obs", "mean_af_obs", "median_af_obs", "iqr_af_obs",
            "wmean_af_obs", "wvar_af_obs", "n_obs", "sum_coverage"
        ])
    else:
        g = d.groupby("mutation", sort=True, observed=True)

        var_unw   = g["af_obs"].var(ddof=1).fillna(0.0).rename("var_af_obs")
        mean_unw  = g["af_obs"].mean().rename("mean_af_obs")
        med_unw   = g["af_obs"].median().rename("median_af_obs")
        q25       = g["af_obs"].quantile(0.25)
        q75       = g["af_obs"].quantile(0.75)
        iqr_unw   = (q75 - q25).rename("iqr_af_obs")
        n_obs     = g.size().rename("n_obs").astype(int)

        tmp = d.assign(
            w  = d["coverage"].astype(float),
            wx = d["coverage"].astype(float) * d["af_obs"].astype(float),
            wx2= d["coverage"].astype(float) * (d["af_obs"].astype(float) ** 2),
        )
        sum_w  = tmp.groupby("mutation", observed=True)["w"].sum().rename("sum_w")
        sum_wx = tmp.groupby("mutation", observed=True)["wx"].sum()
        sum_wx2= tmp.groupby("mutation", observed=True)["wx2"].sum()

        wmean = (sum_wx / sum_w.replace(0, np.nan)).fillna(0.0).rename("wmean_af_obs")
        m2    = (sum_wx2 / sum_w.replace(0, np.nan)).rename("m2")
        wvar  = (m2 - wmean**2).clip(lower=0.0).fillna(0.0).rename("wvar_af_obs")

        sum_cov = sum_w.rename("sum_coverage").astype(float)

        var_tbl = (
            pd.concat([var_unw, mean_unw, med_unw, iqr_unw, wmean, wvar, n_obs, sum_cov], axis=1)
              .reset_index()
              .rename(columns={"index": "mutation"})
              .sort_values(["var_af_obs", "mutation"], ascending=[False, True], kind="mergesort")
              .reset_index(drop=True)
        )

    # -------- MISSINGNESS on full site×date grid --------
    have_grid = {"site_id", "date"}.issubset(d.columns)
    if (not have_grid) or d.empty:
        miss_tbl = pd.DataFrame({
            "mutation": sorted(d["mutation"].unique().tolist()),
            "missingness_rate": 1.0,
            "present_rate": 0.0,
            "present_count": 0,
            "grid_size": 0,
            "n_sites": 0,
            "n_dates": 0,
        })
        return var_tbl, miss_tbl

    # Clean grid keys
    d["site_id"] = d["site_id"].astype(str)
    d["date"]    = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["date"]).copy()

    sites = sorted(d["site_id"].unique().tolist())
    dates = sorted(d["date"].unique().tolist())
    n_sites, n_dates = len(sites), len(dates)
    grid_size = n_sites * n_dates

    if grid_size == 0:
        miss_tbl = pd.DataFrame({
            "mutation": sorted(d["mutation"].unique().tolist()),
            "missingness_rate": 1.0,
            "present_rate": 0.0,
            "present_count": 0,
            "grid_size": 0,
            "n_sites": n_sites,
            "n_dates": n_dates,
        })
        return var_tbl, miss_tbl

    # Presence if any row at (site,date,mutation) has coverage ≥ min_coverage
    present = (
        d.assign(present=(d["coverage"] >= int(min_coverage)).astype(int))
         .groupby(["site_id", "date", "mutation"], observed=True)["present"]
         .max()
         .unstack("mutation")
    )

    # Reindex to full grid (absent=0)
    full_idx = pd.MultiIndex.from_product([sites, dates], names=["site_id", "date"])
    present = present.reindex(full_idx).fillna(0).astype(int)

    # Per-mutation stats
    present_count   = present.sum(axis=0).astype(int).rename("present_count")
    present_rate    = present.mean(axis=0).astype(float).rename("present_rate")
    missingness_rate= (1.0 - present_rate).rename("missingness_rate")

    miss_tbl = (
        pd.concat([missingness_rate, present_rate, present_count], axis=1)
          .reset_index()
          .rename(columns={"index": "mutation"})
    )
    miss_tbl["grid_size"] = int(grid_size)
    miss_tbl["n_sites"]   = int(n_sites)
    miss_tbl["n_dates"]   = int(n_dates)

    miss_tbl = (
        miss_tbl.sort_values(["missingness_rate", "mutation"], ascending=[False, True], kind="mergesort")
                .reset_index(drop=True)
    )

    return var_tbl, miss_tbl
def beta_binomial_overdisp_moments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beta–Binomial overdispersion per mutation using a coverage-aware method-of-moments.

    Output columns (unchanged):
      ['mutation', 'p_hat', 'n_bar', 'var_y', 'phi_mom',
       'rho_mom', 'kappa_mom', 'alpha_mom', 'beta_mom',
       's2_prop', 's2_prop_theory_min', 's2_prop_theory_max',
       'n_reps', 'sum_counts', 'sum_coverage', 'solver_status']
    """
    # ---- Defensive copy & normalization ----
    d = df.copy()

    if "mutation" not in d.columns:
        d["mutation"] = "NA"
    d["mutation"] = d["mutation"].astype(str)

    # coverage
    d["coverage"] = pd.to_numeric(d.get("coverage", 0), errors="coerce").fillna(0.0).astype(float)
    d.loc[d["coverage"] < 0, "coverage"] = 0.0

    # counts (fallback from af_obs if absent or all invalid)
    if "count" in d.columns:
        d["count"] = pd.to_numeric(d["count"], errors="coerce").fillna(0.0).astype(float)
    else:
        d["count"] = 0.0

    # If counts are missing or non-finite everywhere, derive from af_obs * coverage row-wise when possible
    if "af_obs" in d.columns and (("count" not in df.columns) or ~np.isfinite(d["count"]).any()):
        af = pd.to_numeric(d["af_obs"], errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)
        d["count"] = af * d["coverage"]

    # Clip counts to [0, coverage]
    d["count"] = d["count"].clip(lower=0.0)
    bad = d["count"] > d["coverage"]
    if bad.any():
        d.loc[bad, "count"] = d.loc[bad, "coverage"]

    rows = []
    KAPPA_MAX = 1e12
    EPS = 1e-12

    # ---- Internal solver: monotone bisection for κ ≥ 0 ----
    def _solve_kappa(mu: float, n_vec: np.ndarray, s2_prop: float) -> Tuple[float, str]:
        """
        Solve mean_i[ μ(1−μ) * (n_i + κ) / ((1 + κ) n_i) ] = s2_prop  for κ ≥ 0.
        Returns (kappa, status) with status ∈ {'ok','at_lower','at_upper','degenerate','insufficient'}.
        """
        if not (0.0 < mu < 1.0):
            return (np.nan, "degenerate")
        if not np.isfinite(s2_prop):
            return (np.nan, "insufficient")

        inv_n = 1.0 / np.maximum(n_vec, 1.0)
        s2_max = mu * (1.0 - mu)                        # κ = 0 (max overdispersion)
        s2_min = mu * (1.0 - mu) * float(np.mean(inv_n))  # κ → ∞ (binomial limit)

        # Boundary checks (with small tolerance)
        if s2_prop >= s2_max - 1e-12:
            return (0.0, "at_lower")
        if s2_prop <= s2_min + 1e-12:
            return (KAPPA_MAX, "at_upper")

        # Strictly decreasing function in κ
        def f(kappa: float) -> float:
            return float(mu * (1.0 - mu) * np.mean((n_vec + kappa) / ((1.0 + kappa) * n_vec)) - s2_prop)

        lo, hi = 0.0, KAPPA_MAX
        flo, fhi = f(lo), f(hi)  # flo > 0, fhi < 0 by construction
        # Guard in case of any numeric pathology
        if not np.isfinite(flo) or not np.isfinite(fhi):
            return (np.nan, "insufficient")

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            val = f(mid)
            if val > 0.0:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) <= 1e-9 * (1.0 + lo + hi):
                break
        kappa = float(0.5 * (lo + hi))
        return (kappa, "ok")

    # ---- Per-mutation loop ----
    for mut, g in d.groupby("mutation", sort=True, observed=True):
        y = g["count"].to_numpy(dtype=float)
        n = g["coverage"].to_numpy(dtype=float)

        mask = np.isfinite(y) & np.isfinite(n) & (n > 0)
        y = y[mask]; n = n[mask]
        n_reps = int(y.size)

        sum_y = float(np.sum(y)) if n_reps else 0.0
        sum_n = float(np.sum(n)) if n_reps else 0.0

        if n_reps < 2 or sum_n <= 0.0:
            rows.append({
                "mutation": mut,
                "p_hat": np.nan,
                "n_bar": float(np.nan),
                "var_y": float(np.nan),
                "phi_mom": float(np.nan),
                "rho_mom": float(np.nan),
                "kappa_mom": float(np.nan),
                "alpha_mom": float(np.nan),
                "beta_mom": float(np.nan),
                "s2_prop": float(np.nan),
                "s2_prop_theory_min": float(np.nan),
                "s2_prop_theory_max": float(np.nan),
                "n_reps": n_reps,
                "sum_counts": sum_y,
                "sum_coverage": sum_n,
                "solver_status": "insufficient",
            })
            continue

        mu    = sum_y / max(sum_n, 1.0)
        n_bar = float(np.mean(n))
        var_y = float(np.var(y, ddof=1))
        z     = y / n
        s2_prop = float(np.var(z, ddof=1))  # sample variance of proportions

        if not (0.0 < mu < 1.0):
            rows.append({
                "mutation": mut,
                "p_hat": float(mu),
                "n_bar": float(n_bar),
                "var_y": float(var_y),
                "phi_mom": float(np.nan),
                "rho_mom": float(np.nan),
                "kappa_mom": float(np.nan),
                "alpha_mom": float(np.nan),
                "beta_mom": float(np.nan),
                "s2_prop": float(s2_prop),
                "s2_prop_theory_min": float(np.nan),
                "s2_prop_theory_max": float(np.nan),
                "n_reps": n_reps,
                "sum_counts": sum_y,
                "sum_coverage": sum_n,
                "solver_status": "degenerate",
            })
            continue

        mean_inv_n = float(np.mean(1.0 / n))
        s2_max = mu * (1.0 - mu)
        s2_min = mu * (1.0 - mu) * mean_inv_n

        kappa, status = _solve_kappa(mu, n, s2_prop)

        if np.isfinite(kappa):
            kappa = float(np.clip(kappa, 0.0, KAPPA_MAX))
            rho   = 1.0 / (1.0 + kappa)
            alpha = mu * kappa
            beta  = (1.0 - mu) * kappa
        else:
            rho = alpha = beta = np.nan

        rows.append({
            "mutation": mut,
            "p_hat": float(mu),
            "n_bar": float(n_bar),
            "var_y": float(var_y),
            "phi_mom": float(kappa),       # legacy name retained (== kappa)
            "rho_mom": float(rho),
            "kappa_mom": float(kappa),
            "alpha_mom": float(alpha),
            "beta_mom": float(beta),
            "s2_prop": float(s2_prop),
            "s2_prop_theory_min": float(s2_min),
            "s2_prop_theory_max": float(s2_max),
            "n_reps": n_reps,
            "sum_counts": sum_y,
            "sum_coverage": sum_n,
            "solver_status": status,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        # Ensure numeric dtypes are floats (not objects) for downstream CSVs
        num_cols = ["p_hat","n_bar","var_y","phi_mom","rho_mom","kappa_mom",
                    "alpha_mom","beta_mom","s2_prop","s2_prop_theory_min",
                    "s2_prop_theory_max","n_reps","sum_counts","sum_coverage"]
        for c in num_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.sort_values(["phi_mom", "mutation"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    return out

# ---------------------------
# Site-level figures
# ---------------------------
def site_variant_panel(
    site_df: pd.DataFrame,
    sigs_df: pd.DataFrame,
    lineages_df: Optional[pd.DataFrame],
    pcfg: PreprocConfig,
    site_id: str,
    mut_to_color: Dict[str, tuple],
    lin_to_color: Dict[str, tuple],
) -> plt.Figure:
    """
    2×2 lineage signature panel (AF over time) with a **dedicated legend row UNDER the figure**.
    Legend is deduplicated across all four panels and baked into the PNG (no overlap/clipping).
    """
    import math
    import matplotlib.dates as mdates

    set_matplotlib_style()

    # ---- lineage -> [mutations] (deterministic) ----
    lin_to_muts: Dict[str, List[str]] = {}
    has_lineage_col = "lineage" in sigs_df.columns
    for _, row in sigs_df.iterrows():
        lin = str(row["lineage"]) if has_lineage_col else "NA"
        mut = str(row["mutation"])
        lin_to_muts.setdefault(lin, []).append(mut)
    for lin in lin_to_muts:
        lin_to_muts[lin] = sorted(set(lin_to_muts[lin]))

    # ---- which lineages appear at this site? (any AF_obs > 0 among their signature muts) ----
    present_lineages: List[Tuple[str, int]] = []
    for lin, muts in lin_to_muts.items():
        present = site_df[(site_df["mutation"].isin(muts)) & (site_df["af_obs"] > 0)]
        if not present.empty:
            present_lineages.append((lin, len(present)))
    present_lineages.sort(key=lambda x: x[1], reverse=True)

    # take top 4; pad to preserve 2×2 grid symmetry
    selected_lineages = [lin for lin, _ in present_lineages[:4]]
    while len(selected_lineages) < 4:
        selected_lineages.append("")

    # ---- common axes limits (dates + AF) ----
    sdf = site_df.dropna(subset=["date"]).copy()
    date_min = pd.to_datetime(sdf["date"].min()) if not sdf.empty else None
    date_max = pd.to_datetime(sdf["date"].max()) if not sdf.empty else None
    if (date_min is not None) and (date_max is not None) and (date_min == date_max):
        date_max = date_min + pd.Timedelta(days=1)  # widen single-day windows

    AF_YMIN, AF_YMAX = 0.0, 1.02

    # ---- Figure with bottom legend row (3 rows: 2 plot rows + 1 legend row) ----
    fig = plt.figure(figsize=(12.4, 9.8))
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[1.0, 1.0, 0.28],  # bottom row reserved for legend
        left=0.07, right=0.99, top=0.90, bottom=0.08, hspace=0.30, wspace=0.10
    )

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])
    leg_ax = fig.add_subplot(gs[2, :]); leg_ax.axis("off")

    # shared date formatter
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    formatter = mdates.ConciseDateFormatter(locator)

    # Optional pretty lineage labels
    if lineages_df is not None and set(["lineage", "label"]).issubset(lineages_df.columns):
        label_map = dict(zip(lineages_df["lineage"].astype(str), lineages_df["label"].astype(str)))
    else:
        label_map = {}

    # global registry: first handle per mutation becomes the legend entry
    legend_handles: Dict[str, object] = {}

    # helper to plot one panel
    def _plot_panel(ax: plt.Axes, lin: str) -> None:
        ax.set_axisbelow(True)
        ax.grid(True, ls=":", alpha=0.25)

        if not lin:
            ax.set_visible(False)
            return

        muts = lin_to_muts.get(lin, [])
        for m in muts:
            sub = (
                site_df.loc[site_df["mutation"] == m, ["date", "af_obs"]]
                       .dropna(subset=["date"])
                       .sort_values("date")
            )
            sub = sub[sub["af_obs"] > 0]
            if sub.empty:
                continue

            color = mut_to_color.get(m, (0.2, 0.2, 0.2, 1.0))
            pts = ax.scatter(
                sub["date"], sub["af_obs"],
                s=20, alpha=0.9, color=color, label=m, rasterized=True, zorder=2
            )
            # Deduplicate legend entries across all panels
            if m not in legend_handles:
                legend_handles[m] = pts

        # left-censor reference line
        ax.axhline(pcfg.left_censor_af, ls="--", lw=1.0, color="black", alpha=0.8)

        # panel title: pretty label if available
        ax.set_title(label_map.get(lin, lin))

        # unified scales and ticks
        ax.set_ylim(AF_YMIN, AF_YMAX)
        if (date_min is not None) and (date_max is not None):
            ax.set_xlim(date_min, date_max)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        ax.set_yticks(np.linspace(0, 1.0, 6))

    # plot into the four panels
    for ax, lin in zip([ax11, ax12, ax21, ax22], selected_lineages):
        _plot_panel(ax, lin)

    # shared axis labels (bottom row only for x)
    ax21.set_xlabel("Date")
    ax22.set_xlabel("Date")
    ax11.set_ylabel("Allele frequency (AF)")
    ax21.set_ylabel("Allele frequency (AF)")

    # ---- figure-level legend UNDER the panels, framed (uses dedicated axes) ----
    if legend_handles:
        labels = list(legend_handles.keys())
        handles = list(legend_handles.values())

        # reasonable number of columns; scales with label count
        ncol = min(8, max(2, int(math.ceil(len(labels) / 2))))
        leg = leg_ax.legend(
            handles, labels,
            loc="center",
            ncol=ncol,
            frameon=True, fancybox=True, framealpha=0.95, edgecolor="0.6",
            borderpad=0.6, columnspacing=1.2, handletextpad=0.6, scatterpoints=1,
            fontsize=8,
        )
        leg.set_title("Mutations", prop={"size": 9})
        try:
            leg.get_frame().set_linewidth(0.8)
        except Exception:
            pass
    else:
        # keep bottom row height consistent even without entries
        leg_ax.text(0.5, 0.5, "", ha="center", va="center")

    # Figure title
    fig.suptitle(f"Site {site_id} — Variant panel (lineage × mutations, AF scatter)", fontsize=14)

    # Align y-labels vertically within columns for symmetry (best-effort)
    try:
        fig.align_ylabels([ax11, ax21])
        fig.align_ylabels([ax12, ax22])
    except Exception:
        pass

    return fig
def site_analysis_panel(
    site_df: pd.DataFrame,
    sigs_df: pd.DataFrame,
    bias_loci_df: pd.DataFrame,
    pcfg: PreprocConfig,
    site_id: str,
    mut_to_color: Dict[str, tuple],
) -> plt.Figure:
    """
    Site analysis (2×2) with **per-panel legends centered UNDER each subplot** in framed boxes.
    Layout is a 4×2 GridSpec:

        Row 0:  A plot     |  B plot
        Row 1:  A legend   |  B legend
        Row 2:  C plot     |  D plot
        Row 3:  C legend   |  D legend

    This guarantees legends are *under* each panel with ample spacing in saved PNG/PDFs.
    """
    import math
    import numpy as np
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    set_matplotlib_style()

    # -----------------------
    # Shared date/AF limits
    # -----------------------
    sdf = site_df.dropna(subset=["date"]).copy()
    date_min = pd.to_datetime(sdf["date"].min()) if not sdf.empty else None
    date_max = pd.to_datetime(sdf["date"].max()) if not sdf.empty else None
    if (date_min is not None) and (date_max is not None) and (date_min == date_max):
        date_max = date_min + pd.Timedelta(days=1)

    AF_YMIN, AF_YMAX = 0.0, 1.05
    date_locator  = mdates.AutoDateLocator(minticks=3, maxticks=6)
    date_formatter= mdates.ConciseDateFormatter(date_locator)

    # -----------------------
    # Legend sizing helpers
    # -----------------------
    def _cols_for(n: int, cap: int) -> int:
        return min(cap, max(2, int(math.ceil(max(n, 1) / 2))))

    def _rows_for(n: int, ncol: int) -> int:
        return 0 if n <= 0 else int(math.ceil(n / max(1, ncol)))

    def _leg_height(rows: int) -> float:
        # Fraction of a plot row reserved for the legend axes
        return 0.22 + 0.10 * max(rows - 1, 0) if rows > 0 else 0.14

    # -----------------------
    # Pre-compute expected legend loads
    # -----------------------
    # (A) — AF over time: how many mutation labels (AF>0)?
    muts_present = (
        site_df.loc[site_df["af_obs"] > 0, "mutation"]
               .astype(str).sort_values().unique().tolist()
    )
    nA = len(muts_present)
    ncolA = _cols_for(nA, 8)
    rowsA = _rows_for(nA, ncolA)

    # (B) — Alt/Ref: 'Rows' + AF lines + iso-coverage lines (up to 4)
    base_fs = [0.5, 0.1]
    if 0.0 < float(pcfg.left_censor_af) < 1.0:
        base_fs.append(float(pcfg.left_censor_af))
    base_fs = sorted(set(f for f in base_fs if 0.0 < f < 1.0), reverse=True)
    n_iso_af  = len(base_fs)
    n_iso_cov = 4  # upper bound; actual lines may be fewer
    nB = 1 + n_iso_af + n_iso_cov
    ncolB = _cols_for(nB, 8)
    rowsB = _rows_for(nB, ncolB)

    # (C) — Coverage bias: flagged vs other
    flagged_muts = set(
        bias_loci_df.loc[
            (bias_loci_df["flag_dropout"] | bias_loci_df["flag_ref_bias"]),
            "mutation"
        ].astype(str)
    )
    nC = 2 if flagged_muts else 1
    ncolC = nC
    rowsC = 1 if nC else 0

    # (D) — Lineage presence: how many lineages may appear?
    lineage_to_weights: Dict[str, Dict[str, float]] = {}
    for _, row in sigs_df.iterrows():
        lineage_to_weights.setdefault(str(row["lineage"]), {})[str(row["mutation"])] = float(row["weight"])
    muts_at_site = set(site_df["mutation"].astype(str))
    lin_present  = sorted([lin for lin, mw in lineage_to_weights.items() if any(m in muts_at_site for m in mw)])
    nD    = len(lin_present)
    ncolD = _cols_for(nD, 8)
    rowsD = _rows_for(nD, ncolD)

    # Compute legend row heights
    top_leg_h = _leg_height(max(rowsA, rowsB))
    bot_leg_h = _leg_height(max(rowsC, rowsD))

    # -----------------------
    # Figure & Grid (4 rows × 2 cols)
    # -----------------------
    fig_h = 10.0 + 0.9 * (max(rowsA, rowsB) + max(rowsC, rowsD))
    fig_h = max(10.0, min(20.0, fig_h))

    fig = plt.figure(figsize=(12.8, fig_h))
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        height_ratios=[1.0, top_leg_h, 1.0, bot_leg_h],
        left=0.07, right=0.99, top=0.90, bottom=0.08,
        hspace=0.42, wspace=0.16  # airy spacing
    )

    # Main axes
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[2, 0])
    axD = fig.add_subplot(gs[2, 1])
    # Legend axes (dedicated)
    legA = fig.add_subplot(gs[1, 0]); legA.axis("off")
    legB = fig.add_subplot(gs[1, 1]); legB.axis("off")
    legC = fig.add_subplot(gs[3, 0]); legC.axis("off")
    legD = fig.add_subplot(gs[3, 1]); legD.axis("off")

    # -----------------------
    # Panel A — AF over time
    # -----------------------
    axA.set_axisbelow(True)
    axA.grid(True, ls=":", alpha=0.28)
    handles_A: List[object] = []
    labels_A:  List[str]    = []

    for m in sorted(site_df["mutation"].astype(str).unique()):
        sub = (
            site_df.loc[site_df["mutation"] == m, ["date", "af_obs"]]
                   .dropna(subset=["date"])
                   .sort_values("date")
        )
        sub = sub[sub["af_obs"] > 0]
        if sub.empty:
            continue
        color = mut_to_color.get(m, (0.2, 0.2, 0.2, 1.0))
        axA.scatter(sub["date"], sub["af_obs"], s=20, alpha=0.9,
                    color=color, label=m, rasterized=True, zorder=2)
        labels_A.append(m)
        handles_A.append(Line2D([0], [0], marker='o', linestyle='None',
                                markersize=6, markerfacecolor=color,
                                markeredgecolor='none', label=m))

    axA.axhline(pcfg.left_censor_af, ls="--", color="black", lw=1.0)
    axA.set_title("AF over time (all mutations)")
    axA.set_ylabel("Allele frequency")
    axA.set_ylim(AF_YMIN, AF_YMAX)
    if (date_min is not None) and (date_max is not None):
        axA.set_xlim(date_min, date_max)
        axA.xaxis.set_major_locator(date_locator)
        axA.xaxis.set_major_formatter(date_formatter)

    # Legend A (cap)
    if labels_A:
        MAX_A = 36
        if len(labels_A) > MAX_A:
            handles_A = handles_A[:MAX_A]
            labels_A  = labels_A[:MAX_A] + [f"+{len(labels_A)-MAX_A} more"]
        ncols = _cols_for(len(labels_A), 10)
        legA.legend(handles_A, labels_A, loc="center", ncol=ncols,
                    frameon=True, fancybox=True, framealpha=0.96, edgecolor="0.6",
                    fontsize=8.2, handletextpad=0.7, columnspacing=1.4, borderpad=0.7
        ).set_title("Mutations", prop={"size": 9.2})

    # -----------------------
    # Panel B — Alt vs Ref with guides
    # -----------------------
    axB.set_axisbelow(True)
    axB.grid(True, ls=":", alpha=0.28)

    tmp = site_df.copy()
    tmp["ref_count"] = np.maximum(tmp["coverage"] - tmp["count"], 0.0)
    alt = tmp["count"].to_numpy(dtype=float)
    ref = tmp["ref_count"].to_numpy(dtype=float)
    cov = tmp["coverage"].to_numpy(dtype=float)

    cov_valid = cov[np.isfinite(cov) & (cov >= 0)]
    m99 = float(np.percentile(cov_valid, 99.5)) if cov_valid.size else 1.0
    lim = max(1.0, m99) * 1.05

    scB = axB.scatter(alt, ref, s=12, alpha=0.55, label="Rows", rasterized=True)
    xg  = np.linspace(0, lim, 512)

    # AF lines
    line_handles, line_labels = [], []
    for f, style in zip(base_fs, ["--", ":", "-."][:len(base_fs)]):
        slope = (1.0 / f) - 1.0
        (ln,) = axB.plot(xg, slope * xg, style, lw=1.2, color="black", label=f"{f*100:.1f}% AF")
        line_handles.append(ln); line_labels.append(f"{f*100:.1f}% AF")

    # Iso-coverage lines (quantiles actually present)
    cov_lines, cov_labels = [], []
    if cov_valid.size:
        qs = np.quantile(cov_valid, [0.25, 0.5, 0.75, 0.9])
        for c, style in zip(qs, ["-", "--", ":", "-."]):
            if c <= 0:
                continue
            xs = np.linspace(0, min(lim, c), 2)
            (ln,) = axB.plot(xs, c - xs, style, lw=1.0, color="grey", alpha=0.9, label=f"Coverage = {c:.0f}")
            cov_lines.append(ln); cov_labels.append(f"Coverage = {c:.0f}")

    axB.set_title("Alt vs Ref counts")
    axB.set_xlabel("Alt count")
    axB.set_ylabel("Ref count")
    axB.set_xlim(0, lim); axB.set_ylim(0, lim)
    axB.set_aspect("equal", adjustable="box")

    # Legend B
    labels_B  = ["Rows"] + line_labels + cov_labels
    handles_B = [scB] + line_handles + cov_lines
    if labels_B:
        MAX_B = 28
        if len(labels_B) > MAX_B:
            handles_B = handles_B[:MAX_B]
            labels_B  = labels_B[:MAX_B] + [f"+{len(labels_B)-MAX_B} more"]
        ncols = _cols_for(len(labels_B), 8)
        legB.legend(
            handles_B, labels_B, loc="center", ncol=ncols,
            frameon=True, fancybox=True, framealpha=0.96, edgecolor="0.6",
            fontsize=8.2, handletextpad=0.7, columnspacing=1.3, borderpad=0.7
        ).set_title("Guides", prop={"size": 9.2})

    # -----------------------
    # Panel C — Coverage per mutation + flagged highlighting
    # -----------------------
    axC.set_axisbelow(True)
    axC.grid(True, axis="y", ls=":", alpha=0.28)

    order = (
        site_df.groupby("mutation")["coverage"]
               .median().sort_values().index.tolist()
    )
    if HAS_SEABORN and len(order) > 0:
        sns.boxplot(
            x="mutation", y="coverage",
            data=site_df, order=order, ax=axC,
            color="skyblue", showfliers=False
        )
        # overlay points: swarm for small data, strip for larger
        n_points = int(len(site_df)); n_cat = int(len(order))
        avg_per_cat = n_points / max(1, n_cat)
        if avg_per_cat <= 200 and n_cat <= 30:
            sns.swarmplot(
                x="mutation", y="coverage",
                data=site_df, order=order, ax=axC,
                color="black", size=1.8, alpha=0.35
            )
        else:
            sns.stripplot(
                x="mutation", y="coverage",
                data=site_df, order=order, ax=axC,
                color="black", size=1.2, alpha=0.28, jitter=0.25
            )
    else:
        bp = axC.boxplot(
            [site_df.loc[site_df["mutation"] == m, "coverage"] for m in order],
            showfliers=False, patch_artist=True
        )
        colors = categorical_palette(len(order))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        axC.set_xticks(np.arange(1, len(order) + 1))
        axC.set_xticklabels(order)

    axC.set_title("Per-mutation coverage")
    axC.set_ylabel("Coverage")
    rot = 90 if len(order) > 20 else 45
    for lbl in axC.get_xticklabels():
        lbl.set_rotation(rot); lbl.set_horizontalalignment("right")

    # Flagged highlight
    if len(order) > 0:
        xticks = np.arange(len(order))
        for xi, mut in zip(xticks, order):
            if mut in flagged_muts:
                axC.axvspan(xi - 0.5, xi + 0.5, color="mistyrose", alpha=0.6, zorder=0)

    # Legend C
    handles_C, labels_C = [], []
    if len(flagged_muts) > 0:
        handles_C.append(Patch(facecolor="mistyrose", edgecolor="red")); labels_C.append("Flagged locus")
    handles_C.append(Patch(facecolor="skyblue", edgecolor="black")); labels_C.append("Other locus")
    ncols = _cols_for(len(labels_C), 6)
    legC.legend(
        handles_C, labels_C, loc="center", ncol=ncols,
        frameon=True, fancybox=True, framealpha=0.96, edgecolor="0.6",
        fontsize=8.2, handletextpad=0.7, columnspacing=1.3, borderpad=0.7
    ).set_title("Coverage bias", prop={"size": 9.2})

    # -----------------------
    # Panel D — Lineage signature presence (weighted)
    # -----------------------
    axD.set_axisbelow(True)
    axD.grid(True, ls=":", alpha=0.28)

    dates = sorted(site_df["date"].dropna().unique())
    handles_D: List[object] = []
    labels_D:  List[str]    = []

    for lin, mw in sorted(lineage_to_weights.items()):
        series = []
        for dt in dates:
            gdt = site_df[site_df["date"] == dt]
            if gdt.empty:
                series.append(0.0); continue
            total_w = sum(mw.values()) or 1.0
            present_w = 0.0
            for mut, w in mw.items():
                gg = gdt[gdt["mutation"] == mut]
                if gg.empty():
                    continue
                csum = float(gg["coverage"].sum())
                afw  = float((gg["af_obs"] * gg["coverage"]).sum() / max(csum, 1.0))
                if afw > 0:
                    present_w += w
            series.append(present_w / total_w)
        if series and max(series) > 0:
            sc = axD.scatter(dates, series, s=20, alpha=0.9, label=lin, rasterized=True)
            handles_D.append(sc); labels_D.append(lin)

    axD.set_ylabel("Fraction of signature mutations detected")
    axD.set_ylim(0, 1.05)
    axD.set_title("Lineage signature presence (weighted)")
    if (date_min is not None) and (date_max is not None):
        axD.set_xlim(date_min, date_max)
        axD.xaxis.set_major_locator(date_locator)
        axD.xaxis.set_major_formatter(date_formatter)

    axC.set_xlabel("Date"); axD.set_xlabel("Date")

    # Legend D (cap)
    if labels_D:
        MAX_D = 28
        if len(labels_D) > MAX_D:
            handles_D = handles_D[:MAX_D]
            labels_D  = labels_D[:MAX_D] + [f"+{len(labels_D)-MAX_D} more"]
        ncols = _cols_for(len(labels_D), 8)
        legD.legend(
            handles_D, labels_D, loc="center", ncol=ncols,
            frameon=True, fancybox=True, framealpha=0.96, edgecolor="0.6",
            fontsize=8.2, handletextpad=0.7, columnspacing=1.3, borderpad=0.7
        ).set_title("Lineages", prop={"size": 9.2})
    else:
        legD.text(0.5, 0.5, "No lineage signal", ha="center", va="center", fontsize=8.5)

    # -----------------------
    # Final touches
    # -----------------------
    try:
        fig.align_ylabels([axA, axC])
        fig.align_ylabels([axB, axD])
    except Exception:
        pass

    fig.suptitle(f"Site {site_id} — Analysis panel", fontsize=14.5)
    return fig



def site_lineage_index_figure(
    site_df: pd.DataFrame,
    sigs_df: pd.DataFrame,
    lineages_df: Optional[pd.DataFrame],
    site_id: str,
) -> plt.Figure:
    """
    Lineage evolution index (per site) with a framed legend centered *under* the plot.
    The legend row height is sized dynamically so it never overlaps the chart.
    """
    import math
    import matplotlib.dates as mdates

    set_matplotlib_style()

    # ---- Guards ----
    if site_df.empty or sigs_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data for lineage index", ha="center", va="center")
        ax.set_title(f"Site {site_id} – Lineage evolution index (scatter)")
        ax.set_ylabel("Index (0–1)")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        return fig

    # Dates (clean, sorted)
    dates = sorted(pd.to_datetime(site_df["date"].dropna().unique()))
    if not dates:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No dates available", ha="center", va="center")
        ax.set_title(f"Site {site_id} – Lineage evolution index (scatter)")
        ax.set_ylabel("Index (0–1)")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        return fig

    # Build lineage -> {mutation: weight}
    lin_to_mw: Dict[str, Dict[str, float]] = {}
    has_weight = "weight" in sigs_df.columns
    for _, row in sigs_df.iterrows():
        lin = str(row["lineage"])
        mut = str(row["mutation"])
        w = float(row["weight"]) if has_weight else 1.0
        lin_to_mw.setdefault(lin, {})[mut] = w

    # Optional pretty labels
    label_map: Dict[str, str] = {}
    if lineages_df is not None and set(["lineage", "label"]).issubset(lineages_df.columns):
        label_map = dict(
            zip(lineages_df["lineage"].astype(str), lineages_df["label"].astype(str))
        )

    # Stable lineage order and colors
    lineages_sorted = sorted(lin_to_mw.keys())
    colors = categorical_palette(len(lineages_sorted))
    lin_color = {lin: colors[i % len(colors)] for i, lin in enumerate(lineages_sorted)}

    # Pre-compute series per lineage to know legend size before laying out the figure
    series_map: Dict[str, List[float]] = {}
    for lin in lineages_sorted:
        mw = lin_to_mw[lin]
        if not mw:
            continue
        s = []
        for dt in dates:
            gdt = site_df[site_df["date"] == dt]
            if gdt.empty:
                s.append(0.0)
                continue
            num = 0.0
            den = 0.0
            for m, w in mw.items():
                gg = gdt[gdt["mutation"] == m]
                if gg.empty:
                    continue
                csum = float(gg["coverage"].sum())
                afw = float((gg["af_obs"] * gg["coverage"]).sum() / max(csum, 1.0))
                num += w * afw
                den += w
            s.append(num / max(den, 1.0))
        if max(s) > 0:
            series_map[lin] = s

    plotted_lineages = [lin for lin in lineages_sorted if lin in series_map]
    n_legend = len(plotted_lineages)

    # Dynamic legend geometry
    if n_legend > 0:
        ncol = min(6, max(2, int(math.ceil(n_legend / 2))))
        rows = int(math.ceil(n_legend / ncol))
    else:
        ncol, rows = 2, 0

    def _legend_height(rows_: int) -> float:
        # Legend row height as a fraction of a plot row
        return 0.18 + 0.09 * max(rows_ - 1, 0) if rows_ > 0 else 0.12

    leg_h = _legend_height(rows)

    # Figure height scales with legend rows (clamped)
    fig_h = 6.6 + 0.6 * max(rows - 1, 0)
    fig_h = max(6.6, min(12.0, fig_h))

    # ---- Figure & GridSpec: one plot row + one legend row ----
    fig = plt.figure(figsize=(10.8, fig_h))
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.0, leg_h],
        left=0.08, right=0.99, top=0.90, bottom=0.10, hspace=0.30
    )
    ax = fig.add_subplot(gs[0, 0])
    leg_ax = fig.add_subplot(gs[1, 0]); leg_ax.axis("off")

    # ---- Plot the series ----
    ax.set_axisbelow(True)
    ax.grid(True, ls=":", alpha=0.3)

    handles, labels = [], []
    for lin in plotted_lineages:
        s = series_map[lin]
        c = lin_color[lin]
        lbl = label_map.get(lin, lin)
        sc = ax.scatter(dates, s, s=22, alpha=0.9, color=c, label=lbl, rasterized=True)
        ax.plot(dates, s, lw=0.9, alpha=0.9, color=c)
        handles.append(sc)
        labels.append(lbl)

    # Axes, labels, formatting
    date_min, date_max = dates[0], dates[-1]
    if date_min == date_max:
        date_max = date_min + pd.Timedelta(days=1)
    ax.set_xlim(date_min, date_max)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_ylim(0, 1.05)
    ax.set_title(f"Site {site_id} – Lineage evolution index (scatter)")
    ax.set_ylabel("Index (0–1)")

    # ---- Legend under the plot (framed, centered) ----
    if labels:
        leg = leg_ax.legend(
            handles, labels,
            loc="center",
            ncol=ncol,
            frameon=True, fancybox=True, framealpha=0.95, edgecolor="0.6",
            fontsize=9, handletextpad=0.6, columnspacing=1.2, borderpad=0.7,
        )
        leg.set_title("Lineages", prop={"size": 10})
    else:
        leg_ax.text(0.5, 0.5, "No lineage signal detected at this site",
                    ha="center", va="center", fontsize=9)

    fig.suptitle(f"Site {site_id} — Lineage index", fontsize=13)
    return fig

def _extract_config(cfg: Dict) -> PreprocConfig:
    """
    Fully defensive configuration extractor that tolerates any shape (dict, Series, DataFrame).
    """
    import numbers

    def _scalar(x):
        """Return a scalar from any list/array/Series/DataFrame/dict structure."""
        if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
            if len(x):
                return _scalar(x.iloc[0] if isinstance(x, pd.Series) else x[0])
            return np.nan
        if isinstance(x, pd.DataFrame):
            # use the first cell value if single-row/single-col
            if x.size == 1:
                return x.values.flatten()[0]
            if len(x) == 1:
                return _scalar(x.iloc[0].to_dict())
            return np.nan
        if isinstance(x, dict):
            vals = list(x.values())
            return _scalar(vals[0]) if vals else np.nan
        return x

    def _as_int(x, default, min_val=None):
        try:
            v = _scalar(x)
            if isinstance(v, numbers.Real):
                v = int(round(v))
            else:
                v = int(float(v))
        except Exception:
            v = int(default)
        if min_val is not None:
            v = max(min_val, v)
        return v

    def _as_float(x, default, lo=None, hi=None):
        try:
            v = float(_scalar(x))
        except Exception:
            v = float(default)
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v

    pcfg = cfg.get("preprocessing") or {}
    if isinstance(pcfg, pd.Series):
        pcfg = pcfg.to_dict()
    elif isinstance(pcfg, pd.DataFrame) and len(pcfg) == 1:
        pcfg = pcfg.iloc[0].to_dict()

    return PreprocConfig(
        min_coverage=_as_int(pcfg.get("min_coverage", 50), 50, min_val=0),
        left_censor_af=_as_float(pcfg.get("left_censor_af", 0.01), 0.01, lo=0.0, hi=1.0),
        min_alt_count=_as_int(pcfg.get("min_alt_count", 2), 2, min_val=0),
        bias_dropout_quantile=_as_float(pcfg.get("bias_dropout_quantile", 0.10), 0.10, lo=0.0, hi=1.0),
        bias_min_highcov_quantile=_as_float(pcfg.get("bias_min_highcov_quantile", 0.70), 0.70, lo=0.0, hi=1.0),
        bias_af_pos_rate_max=_as_float(pcfg.get("bias_af_pos_rate_max", 0.01), 0.01, lo=0.0, hi=1.0),
        ridgeline_sites_max=_as_int(pcfg.get("ridgeline_sites_max", 15), 15, min_val=1),
        heatmap_mutations_max=_as_int(pcfg.get("heatmap_mutations_max", 200), 200, min_val=1),
        seed=_as_int(pcfg.get("seed", 12345), 12345),
    )
# ---------------------------
# Main stage
# ---------------------------
def run_preprocessing(cfg: Dict, ctx: RunContext) -> Dict:
    """
    Preprocessing stage (clean inputs, compute AF/LOD, diagnostics, figures, and tables).

    This variant **only writes CSV tables and image figures** (e.g., PNG) via
    `ctx.write_table(...)` and `ctx.write_figure(...)`. It **does not write any
    report/markdown/TeX** output. The returned bundle also has `"report": False`
    so upstream code won’t try to render a report.

    If you ever want to enable a short markdown report again, set
    `preprocessing.emit_report: true` in your config.
    """
    import re

    stage_name = "preprocessing"
    pcfg = _extract_config(cfg)
    set_global_seeds(pcfg.seed)

    # Strictly opt-in for reports (default: off)
    emit_report = bool((cfg.get("preprocessing") or {}).get("emit_report", False))

    # ---------------------------
    # Load data (strict) + logging
    # ---------------------------
    data_cfg = cfg.get("data", {})
    path_jahn = data_cfg.get("jahn_like", "data/jahn_like.csv")
    path_sigs = data_cfg.get("signatures", "data/signatures.csv")
    path_lin  = data_cfg.get("lineages",  "data/lineages.csv")

    ctx.log(level="INFO", message="Loading input CSVs",
            stage=stage_name,
            paths={"jahn_like": path_jahn, "signatures": path_sigs, "lineages": path_lin})

    long_df = _load_and_coerce_long(path_jahn)

    # Signatures: be forgiving about 'weight' and 'lineage'
    sigs_df = pd.read_csv(path_sigs)
    if "weight" not in sigs_df.columns:
        sigs_df["weight"] = 1.0
        ctx.log(level="WARN", message="signatures.csv missing 'weight' column; defaulting to 1.0",
                stage=stage_name)
    if "lineage" not in sigs_df.columns:
        sigs_df["lineage"] = "NA"
        ctx.log(level="WARN", message="signatures.csv missing 'lineage' column; set to 'NA'",
                stage=stage_name)
    if "mutation" not in sigs_df.columns:
        raise ValueError("signatures.csv must include a 'mutation' column")

    # Optional lineages metadata
    try:
        lineages_df = pd.read_csv(path_lin)
    except Exception:
        lineages_df = None

    # ---------------------------
    # Schema validations (CSV)
    # ---------------------------
    long_df, val_jahn = _validate_jahn_like_schema(long_df)
    sigs_df, val_sigs = _validate_signatures_schema(sigs_df)
    if lineages_df is not None:
        lineages_df, val_lineages = _validate_lineages_schema(lineages_df)
    else:
        val_lineages = [{
            "table": "lineages",
            "check": "optional_missing",
            "passed": True,
            "details": "no lineages.csv found"
        }]
    schema_validation = pd.DataFrame(val_jahn + val_sigs + val_lineages)
    ctx.write_table("schema_validation", schema_validation)

    # Canonicalize/clean signatures
    sigs_df["mutation"] = sigs_df["mutation"].map(_canonicalize_mutation)
    sigs_df["lineage"] = sigs_df["lineage"].astype(str)
    try:
        sigs_df["weight"] = pd.to_numeric(sigs_df["weight"], errors="coerce").fillna(1.0).astype(float)
    except Exception:
        sigs_df["weight"] = 1.0

    # ---------------------------
    # AF / LOD (KEEP-ALL)
    # ---------------------------
    df_clean, lod_summary = compute_af_censor_filter(
        long_df, pcfg.min_coverage, pcfg.left_censor_af, pcfg.min_alt_count
    )
    filter_stats = df_clean.attrs.get("filter_stats", {})
    ctx.log(level="INFO", message="Left-censoring and coverage thresholds computed (KEEP-ALL)",
            stage=stage_name, filter_stats=filter_stats)
    if not lod_summary.empty:
        lod_summary = lod_summary.sort_values(["site_id", "date"])
    ctx.write_table("lod_summary", lod_summary)

    # ---------------------------
    # Missingness & bias loci
    # ---------------------------
    miss_summary, miss_heat = compute_missingness(df_clean, sigs_df, pcfg.min_coverage)
    ctx.write_table("missingness_summary", miss_summary)

    have_bias = False
    try:
        bias_loci_df = compute_bias_loci(
            df_clean,
            pcfg.min_coverage,
            pcfg.left_censor_af,
            pcfg.bias_dropout_quantile,
            pcfg.bias_min_highcov_quantile,
            pcfg.bias_af_pos_rate_max
        )
        ctx.write_table("bias_loci", bias_loci_df)
        flagged = bias_loci_df[(bias_loci_df["flag_dropout"]) | (bias_loci_df["flag_ref_bias"])]
        ctx.log(level="INFO", message="Flagged bias loci",
                stage=stage_name,
                n_flagged=int(len(flagged)),
                mutations=[str(m) for m in flagged["mutation"].head(50).tolist()])

        expected_cols = {
            "mutation", "median_coverage", "coverage_ratio_to_global",
            "af_pos_rate_highcov", "flag_dropout", "flag_ref_bias",
            "dropout_threshold", "highcov_threshold", "global_median_coverage"
        }
        if expected_cols.issubset(set(bias_loci_df.columns)):
            fig_bias = bias_loci_figure(bias_loci_df)
            ctx.write_figure("bias_loci_diagnostics", fig_bias); plt.close(fig_bias)
            have_bias = True
    except NameError:
        ctx.log(level="ERROR", message="compute_bias_loci is not defined — skipping bias-loci diagnostics.",
                stage=stage_name)
        bias_loci_df = pd.DataFrame()
    except Exception as e:
        ctx.log(level="ERROR", message="compute_bias_loci failed — skipping bias-loci diagnostics.",
                stage=stage_name, error=str(e))
        bias_loci_df = pd.DataFrame()

    # ---------------------------
    # Coverage metrics per site/date (CSV)
    # ---------------------------
    coverage_metrics = (
        df_clean.groupby(["site_id", "date"])["coverage"]
        .agg(
            n_obs="size",
            mean="mean",
            median="median",
            p10=lambda x: float(np.quantile(x, 0.10)),
            p90=lambda x: float(np.quantile(x, 0.90)),
        )
        .reset_index()
        .sort_values(["site_id", "date"])
    )
    ctx.write_table("coverage_metrics", coverage_metrics)

    # ---------------------------
    # Global color maps
    # ---------------------------
    all_mutations = sorted(df_clean["mutation"].astype(str).unique().tolist())
    all_lineages = sorted(sigs_df["lineage"].astype(str).unique().tolist()) if "lineage" in sigs_df.columns else []
    mut_to_color, lin_to_color = _build_global_color_maps(all_mutations, all_lineages)

    # ---------------------------
    # Core figures (IMAGES ONLY)
    # ---------------------------
    cov_fig = coverage_panel_figure(df_clean, pcfg.ridgeline_sites_max)
    ctx.write_figure("coverage_panel", cov_fig); plt.close(cov_fig)

    miss_fig = missingness_heatmap_figure(miss_heat, pcfg.heatmap_mutations_max)
    ctx.write_figure("missingness_heatmap", miss_fig); plt.close(miss_fig)

    # ---------------------------
    # Site-level figures (IMAGES ONLY)
    # ---------------------------
    site_variant_keys: List[str] = []
    site_fig_keys: List[str] = []
    site_evo_keys: List[str] = []

    def _sanitize_key(s: str) -> str:
        return re.sub(r"\W+", "_", str(s)).lower().strip("_")

    for site in sorted(map(str, df_clean["site_id"].unique())):
        sdf = df_clean[df_clean["site_id"].astype(str) == site].copy()

        fig_variant = site_variant_panel(sdf, sigs_df, lineages_df, pcfg, site, mut_to_color, lin_to_color)
        key_variant = f"site_{_sanitize_key(site)}_variant_panel"
        ctx.write_figure(key_variant, fig_variant); plt.close(fig_variant)
        site_variant_keys.append(key_variant)

        fig_site = site_analysis_panel(sdf, sigs_df, bias_loci_df, pcfg, site, mut_to_color)
        key_site = f"site_{_sanitize_key(site)}_analysis"
        ctx.write_figure(key_site, fig_site); plt.close(fig_site)
        site_fig_keys.append(key_site)

        fig_evo = site_lineage_index_figure(sdf, sigs_df, lineages_df, site)
        key_evo = f"site_{_sanitize_key(site)}_evo"
        ctx.write_figure(key_evo, fig_evo); plt.close(fig_evo)
        site_evo_keys.append(key_evo)

    # ---------------------------
    # Global biostatistics (IMAGES ONLY)
    # ---------------------------
    fig_ar = alt_ref_scatter_figure(df_clean, pcfg.left_censor_af, label_suffix=None)
    ctx.write_figure("biostats_alt_vs_ref", fig_ar); plt.close(fig_ar)

    fig_cov_violin = coverage_by_mutation_figure(df_clean)
    ctx.write_figure("biostats_coverage_violin", fig_cov_violin); plt.close(fig_cov_violin)

    fig_ecdf = coverage_ecdf_by_site_figure(df_clean)
    ctx.write_figure("biostats_coverage_ecdf_by_site", fig_ecdf); plt.close(fig_ecdf)

    # ---------------------------
    # Tables: prevalence/entropy, variance/missingness, correlations, overdispersion (CSV)
    # ---------------------------
    prevalence_tbl, entropy_tbl = prevalence_entropy_tables(df_clean)
    if not prevalence_tbl.empty:
        prevalence_tbl = prevalence_tbl.sort_values("prevalence_weighted", ascending=False)
    if not entropy_tbl.empty:
        entropy_tbl = entropy_tbl.sort_values("temporal_entropy_normalized", ascending=False)
    ctx.write_table("mutation_prevalence", prevalence_tbl)
    ctx.write_table("mutation_entropy",   entropy_tbl)

    var_tbl, miss_mut_tbl = mutation_variance_and_missingness(df_clean, pcfg.min_coverage)
    ctx.write_table("mutation_variance",    var_tbl)
    ctx.write_table("mutation_missingness", miss_mut_tbl)

    tmp_global = df_clean.copy()
    tmp_global["ref_count"] = np.maximum(tmp_global["coverage"] - tmp_global["count"], 0)
    a = tmp_global["count"].to_numpy(dtype=float)
    r = tmp_global["ref_count"].to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(r)
    try:
        pear = pearsonr(a[mask], r[mask]) if mask.any() else (np.nan, np.nan)
        spear = spearmanr(a[mask], r[mask]) if mask.any() else (np.nan, np.nan)
        corr_tbl = pd.DataFrame([{
            "pearson_r": float(getattr(pear, "statistic", pear[0])),
            "pearson_p": float(getattr(pear, "pvalue",   pear[1])),
            "spearman_r": float(getattr(spear, "statistic", spear[0])),
            "spearman_p": float(getattr(spear, "pvalue",   spear[1])),
        }])
    except Exception:
        corr_tbl = pd.DataFrame([{
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_r": np.nan, "spearman_p": np.nan
        }])
    ctx.write_table("alt_ref_correlation", corr_tbl)

    bb_phi_tbl = beta_binomial_overdisp_moments(df_clean)
    ctx.write_table("overdispersion_mom", bb_phi_tbl)

    # ---------------------------
    # Feature store outputs (CSV ONLY)
    # ---------------------------
    snv_cols = ["site_id", "date", "sample_id", "mutation",
                "count", "coverage", "af", "af_obs", "af_censored", "lod_threshold"]
    feature_store_snv = (
        df_clean.loc[:, snv_cols]
                .sort_values(["site_id", "date", "sample_id", "mutation"])
    )
    ctx.write_table("feature_store_snv", feature_store_snv)

    sigs_out = sigs_df.loc[:, ["mutation", "lineage", "weight"]].copy()
    ctx.write_table("feature_store_signatures", sigs_out)
    if lineages_df is not None:
        ctx.write_table("feature_store_lineages", lineages_df)

    # ---------------------------
    # NO REPORT OUTPUT
    # ---------------------------
    if emit_report:
        # If you deliberately enable it in config, you can re-add a short report here.
        pass  # intentionally disabled by default

    # ---------------------------
    # Site logs (JSON logs only; no files beyond CSV/figures)
    # ---------------------------
    for site in sorted(map(str, df_clean["site_id"].unique())):
        sdf = df_clean[df_clean["site_id"].astype(str) == site]
        meta = {
            "n_samples": int(sdf["sample_id"].nunique()),
            "n_dates": int(sdf["date"].nunique()),
            "mean_coverage": float(sdf["coverage"].mean()),
            "frac_censored": float(sdf["af_censored"].mean()),
        }
        ctx.log(level="INFO", message="Site summary",
                stage=stage_name, site_id=str(site), meta=meta)

    # ---------------------------
    # Bundle index (report=False to avoid any non-CSV/non-image artifacts)
    # ---------------------------
    figures_list = [
        "coverage_panel",
        "missingness_heatmap",
        "biostats_alt_vs_ref",
        "biostats_coverage_violin",
        "biostats_coverage_ecdf_by_site",
        *site_variant_keys,
        *site_fig_keys,
        *site_evo_keys,
    ]
    if have_bias:
        figures_list.insert(1, "bias_loci_diagnostics")

    bundle = {
        "tables": [
            "schema_validation",
            "lod_summary",
            "missingness_summary",
            "bias_loci" if not bias_loci_df.empty else None,
            "coverage_metrics",
            "mutation_prevalence",
            "mutation_entropy",
            "mutation_variance",
            "mutation_missingness",
            "alt_ref_correlation",
            "overdispersion_mom",
            "feature_store_snv",
            "feature_store_signatures",
        ] + (["feature_store_lineages"] if lineages_df is not None else []),
        "figures": [k for k in figures_list if k is not None],
        "report": False,  # <<— critical: do not emit any report artifact
    }
    return bundle
