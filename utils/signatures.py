"""
Shared construction of the lineage signature matrix S.

S is a binary membership matrix: S[m, L] = 1 if mutation m is carried by lineage L,
and 0 otherwise. A mutation may belong to any number of lineages; there is no
row-mass cap. The signatures table must have one row per (mutation, lineage) pair
and a `weight` column, if present, must equal 1.

Used by stages/likelihood.py and stages/forecast.py so both stages see the same S.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

GLOBAL = "GLOBAL"

logger = logging.getLogger(__name__)


def normalize_mutation(x) -> str:
    """Canonical mutation ID: stripped and upper-cased."""
    return str(x).strip().upper()


def normalize_signatures(sig: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean (mutation, lineage) table with one row per pair.

    Raises ValueError if required columns are missing or if a `weight` column
    contains any value other than 1. Rows with a missing mutation or lineage are
    dropped. Duplicate (mutation, lineage) pairs are collapsed and logged.
    """
    missing = {"mutation", "lineage"} - set(sig.columns)
    if missing:
        raise ValueError(f"signatures table missing columns: {sorted(missing)}")

    s = sig.copy()
    if "weight" in s.columns:
        w = pd.to_numeric(s["weight"], errors="coerce")
        n_bad = int((w != 1.0).sum())
        if n_bad:
            raise ValueError(
                f"signatures 'weight' must be 1 (membership); found {n_bad} rows with other values"
            )

    s = s.dropna(subset=["mutation", "lineage"])
    s["mutation"] = s["mutation"].map(normalize_mutation)
    s["lineage"] = s["lineage"].astype(str).str.strip()
    s = s[(s["mutation"] != "") & (s["lineage"] != "")]

    n_dup = int(s.duplicated(["mutation", "lineage"]).sum())
    if n_dup:
        logger.warning("Collapsing %d duplicate (mutation, lineage) rows", n_dup)
        s = s.drop_duplicates(["mutation", "lineage"])

    return s[["mutation", "lineage"]].reset_index(drop=True)


def build_signature_matrix(
    sig: pd.DataFrame,
    target_mutations: Iterable[str],
    *,
    unmapped_to_global: bool = False,
) -> pd.DataFrame:
    """
    Build S as a DataFrame indexed by normalized mutation, columns = lineages.

    Parameters
    ----------
    sig : signatures table with `mutation`, `lineage`, and optionally `weight` (must be 1).
    target_mutations : mutations to include as rows (order is not preserved; output is sorted).
    unmapped_to_global : if True, mutations with no lineage get GLOBAL = 1 and a GLOBAL
        column is always present. If False, unmapped mutations are dropped and no GLOBAL
        column is added.

    Returns
    -------
    pd.DataFrame of float64 with 0/1 entries.
    """
    s = normalize_signatures(sig)
    s["value"] = 1.0
    targets = pd.Index(sorted({normalize_mutation(m) for m in target_mutations}), name="mutation")

    if s.empty:
        wide = pd.DataFrame(0.0, index=targets, columns=[])
    else:
        wide = s.pivot_table(index="mutation", columns="lineage", values="value",
                             aggfunc="max", fill_value=0.0)
    wide = wide.reindex(index=targets, fill_value=0.0).astype(float)

    empty_lineages = wide.columns[wide.sum(axis=0) == 0.0]
    if len(empty_lineages):
        logger.info("Dropping lineages with no target mutations: %s", list(empty_lineages))
        wide = wide.drop(columns=list(empty_lineages))

    unmapped = wide.sum(axis=1) == 0.0
    n_unmapped = int(unmapped.sum())
    if unmapped_to_global:
        if n_unmapped:
            logger.info("Assigning %d unmapped mutations to GLOBAL", n_unmapped)
        wide[GLOBAL] = unmapped.astype(float).to_numpy()
    else:
        if n_unmapped:
            logger.info("Dropping %d mutations with no lineage signal", n_unmapped)
        wide = wide.loc[~unmapped]

    return wide.sort_index().sort_index(axis=1)
