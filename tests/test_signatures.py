"""Tests for the shared signature matrix builder (utils/signatures.py)."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.signatures import GLOBAL, build_signature_matrix  # noqa: E402


def _sig(rows):
    return pd.DataFrame(rows, columns=["mutation", "lineage", "weight"])


def test_multi_lineage_mutation_has_full_membership_in_each_lineage():
    sig = _sig([
        ("A1T", "L1", 1), ("A1T", "L2", 1),
        ("C2G", "L1", 1),
    ])
    S = build_signature_matrix(sig, ["A1T", "C2G"])
    assert S.loc["A1T", "L1"] == 1.0
    assert S.loc["A1T", "L2"] == 1.0
    assert S.loc["C2G", "L2"] == 0.0


def test_row_sums_are_not_capped_for_multi_lineage_mutations():
    sig = _sig([("A1T", "L1", 1), ("A1T", "L2", 1), ("A1T", "L3", 1)])
    S = build_signature_matrix(sig, ["A1T"])
    assert S.loc["A1T"].sum() == 3.0


def test_duplicate_pairs_collapse_to_one():
    sig = _sig([("A1T", "L1", 1), ("A1T", "L1", 1), ("C2G", "L2", 1)])
    S = build_signature_matrix(sig, ["A1T", "C2G"])
    assert S.loc["A1T", "L1"] == 1.0
    assert S.shape == (2, 2)


def test_case_and_whitespace_in_names_are_normalized():
    sig = _sig([("aa:S:N501Y", " L1 ", 1)])
    S = build_signature_matrix(sig, ["AA:S:N501Y"])
    assert S.loc["AA:S:N501Y", "L1"] == 1.0


def test_unmapped_mutations_dropped_without_global():
    sig = _sig([("A1T", "L1", 1)])
    S = build_signature_matrix(sig, ["A1T", "C2G"], unmapped_to_global=False)
    assert list(S.index) == ["A1T"]
    assert GLOBAL not in S.columns


def test_unmapped_mutations_go_to_global_when_requested():
    sig = _sig([("A1T", "L1", 1)])
    S = build_signature_matrix(sig, ["A1T", "C2G"], unmapped_to_global=True)
    assert S.loc["C2G", GLOBAL] == 1.0
    assert S.loc["A1T", GLOBAL] == 0.0
    assert S.loc["A1T", "L1"] == 1.0


def test_non_unit_weight_raises():
    sig = _sig([("A1T", "L1", 0.5)])
    with pytest.raises(ValueError, match="weight"):
        build_signature_matrix(sig, ["A1T"])


def test_missing_required_column_raises():
    sig = pd.DataFrame({"mutation": ["A1T"], "weight": [1]})
    with pytest.raises(ValueError, match="lineage"):
        build_signature_matrix(sig, ["A1T"])


def test_lineage_without_target_mutations_is_dropped():
    sig = _sig([("A1T", "L1", 1), ("X9Y", "L_OTHER", 1)])
    S = build_signature_matrix(sig, ["A1T"])
    assert "L_OTHER" not in S.columns


def test_predictions_stay_in_unit_interval_for_random_simplex_theta():
    rng = np.random.default_rng(4821)
    sig = _sig([
        ("A1T", "L1", 1), ("A1T", "L2", 1), ("A1T", "L3", 1),
        ("C2G", "L2", 1), ("G3T", "L1", 1),
    ])
    S = build_signature_matrix(sig, ["A1T", "C2G", "G3T", "T4C"], unmapped_to_global=True).values
    for _ in range(200):
        theta = rng.dirichlet(np.ones(S.shape[1]))
        p = S @ theta
        assert np.all(p >= -1e-12) and np.all(p <= 1 + 1e-12)
