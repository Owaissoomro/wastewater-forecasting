"""Tests for the Stage 2 figure script (scripts/make_stage2_figure.py)."""

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_stage2_figure.py"
_spec = importlib.util.spec_from_file_location("make_stage2_figure", SCRIPT)
fig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig)

SITE = "S1"
DATE = "2021-11-30"


def _toy():
    S = pd.DataFrame(
        {"L1": [1.0, 1.0, 1.0, 0.0], "L2": [0.0, 0.0, 0.0, 1.0], "GLOBAL": [0.0, 0.0, 0.0, 0.0]},
        index=["M1", "M2", "M3", "M4"],
    )
    r = pd.DataFrame(
        {"pred_af": [0.1, 0.5, 0.9, 0.2], "coverage": [10.0, 50.0, 30.0, 40.0]},
        index=["M1", "M2", "M3", "M4"],
    )
    priors = pd.DataFrame(
        {"mu": [0.1, 0.4, 0.6, 0.2], "kappa": [50.0, 40.0, 30.0, 20.0]},
        index=["M1", "M2", "M3", "M4"],
    )
    lev = pd.DataFrame({
        "site_id": [SITE] * 4, "date": [DATE] * 4,
        "mutation": ["M1", "M2", "M3", "M4"], "leverage": [0.1, 0.9, 0.5, 0.3],
    })
    return S, r, priors, lev


def _mutations(rows):
    return [m for _, m in rows]


def test_top_leverage_picks_highest_leverage_per_lineage():
    S, r, priors, lev = _toy()
    rows = fig.select_mutations(S, r, lev, priors, SITE, DATE, "top_leverage", 2, [])
    assert _mutations(rows) == ["M2", "M3", "M4"]


def test_highest_coverage_orders_by_coverage():
    S, r, priors, lev = _toy()
    rows = fig.select_mutations(S, r, lev, priors, SITE, DATE, "highest_coverage", 2, [])
    assert _mutations(rows) == ["M2", "M3", "M4"]


def test_best_and_worst_predicted_use_absolute_error_of_mu_vs_implied():
    S, r, priors, lev = _toy()
    # |mu - pred|: M1 0.0, M2 0.1, M3 0.3, M4 0.0 (M4 is in L2 alone)
    best = fig.select_mutations(S, r, lev, priors, SITE, DATE, "best_predicted", 1, [])
    worst = fig.select_mutations(S, r, lev, priors, SITE, DATE, "worst_predicted", 1, [])
    assert _mutations(best) == ["M1", "M4"]
    assert _mutations(worst) == ["M3", "M4"]


def test_manual_selection_assigns_each_mutation_to_first_lineage():
    S, r, priors, lev = _toy()
    rows = fig.select_mutations(S, r, lev, priors, SITE, DATE, "manual", 1, ["m3", "M4"])
    assert rows == [("L1", "M3"), ("L2", "M4")]


def test_manual_selection_rejects_unknown_mutation():
    S, r, priors, lev = _toy()
    with pytest.raises(ValueError, match="not in signature matrix"):
        fig.select_mutations(S, r, lev, priors, SITE, DATE, "manual", 1, ["ZZZ"])


def test_unknown_selection_mode_raises():
    S, r, priors, lev = _toy()
    with pytest.raises(ValueError, match="Unknown SELECTION"):
        fig.select_mutations(S, r, lev, priors, SITE, DATE, "nope", 1, [])


def test_pseudo_counts_sum_to_kappa():
    alt, ref = fig.pseudo_counts(0.44, 55.0)
    assert np.isclose(alt, 24.2)
    assert np.isclose(alt + ref, 55.0)


def test_validate_rejects_theta_off_simplex():
    S, _, priors, _ = _toy()
    theta = pd.Series([0.5, 0.2], index=["L1", "L2"])
    with pytest.raises(ValueError, match="simplex"):
        fig.validate(theta, S[["L1", "L2"]], priors)


def test_validate_rejects_non_binary_signature():
    S, _, priors, _ = _toy()
    S = S.copy()
    S.loc["M1", "L1"] = 0.5
    theta = pd.Series([0.6, 0.4], index=["L1", "L2"])
    with pytest.raises(ValueError, match="binary"):
        fig.validate(theta, S[["L1", "L2"]], priors)


def test_validate_accepts_valid_inputs():
    S, _, priors, _ = _toy()
    theta = pd.Series([0.6, 0.4], index=["L1", "L2"])
    fig.validate(theta, S[["L1", "L2"]], priors)


def test_priors_reader_rejects_repeated_header_lines(tmp_path):
    p = tmp_path / "priors_hyperparams.csv"
    p.write_text("mutation,mu,kappa\nmutation,mu,kappa\nM1,0.1,50\n")
    with pytest.raises(ValueError, match="repeated header"):
        fig.read_priors_hyperparams(str(p))


def test_priors_reader_rejects_duplicated_mutations(tmp_path):
    p = tmp_path / "priors_hyperparams.csv"
    p.write_text("mutation,mu,kappa\nM1,0.1,50\nM1,0.2,40\n")
    with pytest.raises(ValueError, match="duplicated"):
        fig.read_priors_hyperparams(str(p))


def test_priors_reader_reads_clean_file(tmp_path):
    p = tmp_path / "priors_hyperparams.csv"
    p.write_text("mutation,mu,kappa\n aa:m1 ,0.1,50\nM2,0.3,20\n")
    df = fig.read_priors_hyperparams(str(p))
    assert list(df.index) == ["AA:M1", "M2"]
    assert np.allclose(df["mu"], [0.1, 0.3])


def test_entropy_date_default_picks_most_even_composition():
    theta = pd.DataFrame({
        "site_id": [SITE] * 6,
        "date": ["2021-01-01"] * 3 + ["2021-01-02"] * 3,
        "lineage": ["L1", "L2", "L3"] * 2,
        "theta": [0.98, 0.01, 0.01, 0.34, 0.33, 0.33],
    })
    res = pd.DataFrame({
        "site_id": [SITE, SITE], "date": ["2021-01-01", "2021-01-02"], "coverage": [10.0, 20.0],
    })
    site, date = fig.resolve_site_date(theta, res, None, None)
    assert site == SITE
    assert date == "2021-01-02"


def test_site_default_uses_total_coverage_across_dates():
    theta = pd.DataFrame({
        "site_id": ["A", "B"], "date": ["2021-01-01", "2021-01-01"],
        "lineage": ["L1", "L1"], "theta": [1.0, 1.0],
    })
    res = pd.DataFrame({
        "site_id": ["A", "A", "B"], "date": ["2021-01-01", "2021-01-02", "2021-01-01"],
        "coverage": [30.0, 30.0, 50.0],
    })
    site, _ = fig.resolve_site_date(theta, res, None, "2021-01-01")
    assert site == "A"
