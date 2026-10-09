"""
Regenerate the Stage 2 evidence figure from likelihood-stage outputs.

Panels
  A  Beta(mu*kappa, (1-mu)*kappa) prior density per mutation; dot at mu (Stage 1 mean)
  B  Pseudo-counts: alternate = mu*kappa, reference = (1-mu)*kappa, total = kappa
  C  Binary signature matrix S (mutation x lineage), theta-hat per lineage, p = S theta
  D  Stage 1 mean mu (open circle) vs implied frequency (S theta-hat)_m (diamond)
  Bottom  Worked prior-to-pseudo-count example and the jointly estimated composition

Inputs
  <LIKELIHOOD_TABLES>/theta_estimates.csv, signatures_used.csv, residuals.csv,
  mutation_leverage.csv, and <PRIORS_DIR>/priors_hyperparams.csv.

Priors come from priors_hyperparams.csv, the same per-mutation values the likelihood
stage uses when detail_global_timeseries.csv has no site-level rows (it currently has
only GLOBAL rows, so the date-specific override is not applied).

Edit the settings block below and run the file; nothing is read from the command line.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from scipy.stats import beta as beta_dist

# ----------------------------- settings -----------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIKELIHOOD_TABLES = os.path.join(REPO_ROOT, "results", "likelihood", "tables")
PRIORS_DIR = os.path.join(REPO_ROOT, "results", "priors")
OUT_DIR = os.path.join(REPO_ROOT, "results")

SITE = None  # None -> site with the largest total coverage across all dates
DATE = "2021-11-30"  # None -> date with the highest lineage-composition entropy for SITE

# "top_leverage" | "highest_coverage" | "best_predicted" | "worst_predicted" | "manual"
SELECTION = "top_leverage"
N_PER_LINEAGE = 10
MANUAL_MUTATIONS: List[str] = []  # used when SELECTION == "manual"; each goes to its first lineage

LINEAGE_ORDER = ["B.1.1.7", "B.1.351", "P.1", "B.1.617.2", "GLOBAL"]
LINEAGE_LABELS = {
    "B.1.1.7": "Alpha",
    "B.1.351": "Beta",
    "P.1": "Gamma",
    "B.1.617.2": "Delta",
    "GLOBAL": "Unassigned",
}
LINEAGE_COLORS = {
    "B.1.1.7": "#2a6fa8",
    "B.1.351": "#17807a",
    "P.1": "#b8762f",
    "B.1.617.2": "#7d59a8",
    "GLOBAL": "#7f8c8d",
}
PRIOR_GRID = np.linspace(0.0005, 0.9995, 400)
EPS = 1e-6
# --------------------------------------------------------------------


def read_priors_hyperparams(path: str) -> pd.DataFrame:
    """Read per-mutation priors; fail on repeated header lines or repeated mutations."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Priors missing: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"priors_hyperparams.csv is malformed ({exc}). It was likely appended to by "
            "more than one run; regenerate it by rerunning the priors stage."
        ) from exc
    if (df["mutation"].astype(str).str.strip() == "mutation").any():
        raise ValueError(f"priors_hyperparams.csv contains repeated header lines: {path}")
    df["mutation"] = df["mutation"].astype(str).str.strip().str.upper()
    dup = df["mutation"].duplicated()
    if dup.any():
        raise ValueError(
            f"priors_hyperparams.csv has {int(dup.sum())} duplicated mutations; "
            "regenerate it with the priors stage"
        )
    return df[["mutation", "mu", "kappa"]].set_index("mutation")


def load_tables(tables_dir: str, priors_dir: str) -> Dict[str, pd.DataFrame]:
    return {
        "theta": pd.read_csv(os.path.join(tables_dir, "theta_estimates.csv")),
        "sig": pd.read_csv(os.path.join(tables_dir, "signatures_used.csv")).set_index("mutation"),
        "res": pd.read_csv(os.path.join(tables_dir, "residuals.csv")),
        "lev": pd.read_csv(os.path.join(tables_dir, "mutation_leverage.csv")),
        "priors": read_priors_hyperparams(os.path.join(priors_dir, "priors_hyperparams.csv")),
    }


def composition_entropy(p: pd.Series) -> float:
    """Shannon entropy (nats) of a lineage composition; zero-weight lineages are ignored."""
    v = p.to_numpy(float)
    v = v[v > 0]
    return float(-(v * np.log(v)).sum())


def resolve_site_date(theta: pd.DataFrame, res: pd.DataFrame,
                      site: Optional[str], date: Optional[str]) -> Tuple[str, str]:
    """
    Site default: the site with the largest total coverage across all dates.
    Date default: the date whose lineage composition (theta-hat) has the highest entropy
    for that site. Explicit SITE/DATE settings override either default.
    """
    if site is None:
        site = str(res.groupby("site_id")["coverage"].sum().idxmax())
    site = str(site)
    if date is None:
        t = theta[theta["site_id"] == site]
        if t.empty:
            raise ValueError(f"No theta estimates for site {site}")
        ent = t.groupby("date")["theta"].apply(composition_entropy)
        date = str(ent.idxmax())
    d = pd.Timestamp(date)
    if res[pd.to_datetime(res["date"]) == d].empty:
        raise ValueError(f"No residual rows for date {d.date()}")
    return site, d.strftime("%Y-%m-%d")


def build_matrices(tables: Dict[str, pd.DataFrame], site: str, date: str):
    """Return theta (lineage-indexed), S (mutation x lineage), and residual rows for site/date."""
    th = tables["theta"]
    t = th[(th["site_id"] == site) & (th["date"] == date)].set_index("lineage")["theta"]
    lineages = [L for L in LINEAGE_ORDER if L in tables["sig"].columns]
    theta = t.reindex(lineages).astype(float)
    S = tables["sig"][lineages].astype(float)
    res = tables["res"]
    r = res[(res["site_id"] == site) & (pd.to_datetime(res["date"]) == pd.Timestamp(date))]
    r = r.drop_duplicates("mutation").set_index("mutation")
    return theta, S, r


def validate(theta: pd.Series, S: pd.DataFrame, priors: pd.DataFrame) -> None:
    if abs(theta.sum() - 1.0) > 1e-6 or np.any(theta < -1e-8):
        raise ValueError("theta-hat is not on the simplex")
    if not np.isin(S.to_numpy(), [0.0, 1.0]).all():
        raise ValueError("signature matrix S is not binary")
    p = priors.loc[priors.index.intersection(S.index)]
    if not p["mu"].between(0, 1).all():
        raise ValueError("prior mu outside [0, 1]")
    if (p["kappa"] < 0).any():
        raise ValueError("prior kappa is negative")


def select_mutations(
    S: pd.DataFrame,
    r: pd.DataFrame,
    lev: pd.DataFrame,
    priors: pd.DataFrame,
    site: str,
    date: str,
    mode: str,
    n: int,
    manual: List[str],
) -> List[Tuple[str, str]]:
    """Return (lineage, mutation) rows in display order, using the selection mode."""
    pred = r["pred_af"].astype(float)
    mu = priors["mu"].reindex(r.index).astype(float)
    if mode == "manual":
        rows = []
        for m in [x.strip().upper() for x in manual]:
            if m not in S.index:
                raise ValueError(f"Manual mutation not in signature matrix: {m}")
            lin = S.columns[S.loc[m].to_numpy() == 1.0]
            rows.append((lin[0] if len(lin) else "GLOBAL", m))
        return rows

    lev_day = lev[(lev["site_id"] == site) & (pd.to_datetime(lev["date"]) == pd.Timestamp(date))]
    leverage = lev_day.drop_duplicates("mutation").set_index("mutation")["leverage"].astype(float)

    def score(m: str) -> float:
        if mode == "top_leverage":
            return -float(leverage.get(m, np.nan))
        if mode == "highest_coverage":
            return -float(r.loc[m, "coverage"])
        err = abs(float(mu.get(m, np.nan)) - float(pred.get(m, np.nan)))
        if mode == "best_predicted":
            return err
        if mode == "worst_predicted":
            return -err
        raise ValueError(f"Unknown SELECTION: {mode}")

    rows: List[Tuple[str, str]] = []
    for L in S.columns:
        members = [m for m in S.index[S[L] == 1.0] if m in r.index]
        scored = sorted(members, key=lambda m: (score(m), m))
        rows.extend((L, m) for m in scored[:n])
    return rows


def pseudo_counts(mu: float, kappa: float) -> Tuple[float, float]:
    return mu * kappa, (1.0 - mu) * kappa


def density_curve(mu: float, kappa: float) -> np.ndarray:
    a, b = pseudo_counts(mu, kappa)
    y = beta_dist.pdf(PRIOR_GRID, max(a, EPS), max(b, EPS))
    y = np.where(np.isfinite(y), y, 0.0)
    return y / y.max() if y.max() > 0 else y


def plot_figure(
    theta: pd.Series,
    S: pd.DataFrame,
    r: pd.DataFrame,
    priors: pd.DataFrame,
    rows: List[Tuple[str, str]],
    site: str,
    date: str,
    out_path: str,
) -> None:
    lineages = list(S.columns)
    mutations = [m for _, m in rows]
    mu = priors["mu"].reindex(mutations).to_numpy(float)
    kap = priors["kappa"].reindex(mutations).to_numpy(float)
    implied = r.loc[mutations, "pred_af"].to_numpy(float)

    # y positions with gaps between lineage groups
    ys, y, prev = [], 0.0, None
    for L, _ in rows:
        if prev is not None and L != prev:
            y += 0.7
        ys.append(y)
        y += 1.0
        prev = L
    ys = np.array(ys)
    ymax = ys.max() + 0.7

    fig = plt.figure(figsize=(16, 11.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[4.6, 1.5], width_ratios=[1.7, 1.1, 1.5, 1.4],
                          hspace=0.42, wspace=0.14, left=0.11, right=0.97, top=0.74, bottom=0.06)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    axC = fig.add_subplot(gs[0, 2], sharey=axA)
    axD = fig.add_subplot(gs[0, 3], sharey=axA)

    fig.text(0.07, 0.965, "Stage 2 | Joint lineage estimation", fontsize=20, weight="bold", color="#1e2d3d")
    fig.text(0.07, 0.93, f"{site}, {date}. One shared composition across all contributing mutations.",
             fontsize=12, color="#5b6b7b")
    # Panel titles and subtitles sit in figure coordinates above the axes.
    for ax, title, sub in [
        (axA, "A  Mutation predictions", r"Beta($\mu_m\kappa_m$, $(1-\mu_m)\kappa_m$)"),
        (axB, "B  Pseudo-counts", r"Alternate + reference"),
        (axC, "C  Shared signature fit", r"One fitted $\hat{\theta}$ for every row"),
        (axD, "D  Mutation agreement", r"Compare $\mu_m$ with $[S\hat{\theta}]_m$"),
    ]:
        x0 = ax.get_position().x0
        fig.text(x0, 0.875, title, fontsize=13, weight="bold", color="#1e2d3d")
        fig.text(x0, 0.85, sub, fontsize=10, color="#5b6b7b")

    # Panel A
    for yi, (L, m), k_, m_ in zip(ys, rows, kap, mu):
        c = LINEAGE_COLORS[L]
        dens = density_curve(m_, k_)
        axA.fill_between(PRIOR_GRID, yi, yi - 0.85 * dens, color=c, alpha=0.18, lw=0)
        axA.plot(PRIOR_GRID, yi - 0.85 * dens, color=c, lw=1.6)
        axA.plot([m_], [yi], "o", color=c, ms=5)
    axA.set_yticks(ys)
    axA.set_yticklabels([f"{m}" for _, m in rows], fontsize=10)
    axA.set_ylim(ymax, -1.0)
    axA.set_xlim(0, 1)
    axA.set_xticks([0, 0.25, 0.5, 0.75, 1])
    axA.set_xlabel("Mutation frequency")

    # Panel B
    alt, ref = zip(*(pseudo_counts(a, b) for a, b in zip(mu, kap)))
    alt, ref = np.array(alt), np.array(ref)
    colors = [LINEAGE_COLORS[L] for L, _ in rows]
    axB.barh(ys, alt, height=0.6, color=colors)
    axB.barh(ys, ref, left=alt, height=0.6, color="#cfd8e0")
    for yi, k_ in zip(ys, kap):
        axB.text(float(kap.max()) * 1.04, yi, f"{k_:.0f}", va="center", fontsize=10, color="#3c4a5a")
    axB.set_xlim(0, float(kap.max()) * 1.12)
    axB.set_xlabel("Pseudo-count support")
    axB.tick_params(axis="y", left=False, labelleft=False)

    # Panel C
    axC.set_xlim(-0.5, len(lineages) - 0.5)
    hdr_tr = blended_transform_factory(axC.transData, fig.transFigure)
    for j, L in enumerate(lineages):
        axC.text(j, 0.815, LINEAGE_LABELS.get(L, L), ha="center", fontsize=10, weight="bold",
                 color=LINEAGE_COLORS.get(L, "#333"), transform=hdr_tr)
        axC.text(j, 0.782, f"{100 * theta[L]:.1f}%", ha="center", fontsize=12, weight="bold",
                 color=LINEAGE_COLORS.get(L, "#333"), transform=hdr_tr)
    for yi, (L, m) in zip(ys, rows):
        for j, LL in enumerate(lineages):
            v = S.loc[m, LL]
            if v == 1.0:
                axC.add_patch(Rectangle((j - 0.4, yi - 0.34), 0.8, 0.68,
                                        color=LINEAGE_COLORS[LL]))
                axC.text(j, yi, "1", ha="center", va="center", color="white", fontsize=11, weight="bold")
            else:
                axC.text(j, yi, "0", ha="center", va="center", color="#b0bac4", fontsize=10)
    axC.set_xticks([])
    axC.tick_params(axis="y", left=False, labelleft=False)

    # Panel D
    axD.set_xlim(0, 1)
    for yi, (L, m), mu_i, imp in zip(ys, rows, mu, implied):
        c = LINEAGE_COLORS[L]
        axD.plot([mu_i, imp], [yi, yi], color=c, lw=1, ls=":", alpha=0.7)
        axD.plot([mu_i], [yi], "o", mfc="white", mec=c, mew=1.8, ms=7)
        axD.plot([imp], [yi], "D", color=c, ms=7)
    axD.set_xticks([0, 0.25, 0.5, 0.75, 1])
    axD.set_xlabel("Mutation frequency")
    axD.tick_params(axis="y", left=False, labelleft=False)

    # Bottom-left: worked example for the first row
    L0, m0 = rows[0]
    a0, b0 = pseudo_counts(mu[0], kap[0])
    axL = fig.add_subplot(gs[1, :2])
    axL.axis("off")
    axL.text(0, 0.92, "PRIOR INFORMATION AS PSEUDO-COUNTS", fontsize=11, weight="bold", color="#5b6b7b",
             transform=axL.transAxes)
    axL.text(0, 0.68, rf"{m0}:   $\mu = {mu[0]:.3f}$,    $\kappa = {kap[0]:.0f}$",
             fontsize=13, transform=axL.transAxes)
    frac = a0 / kap[0] if kap[0] else 0.0
    axL.add_patch(Rectangle((0, 0.28), frac, 0.2, color=LINEAGE_COLORS[L0],
                            transform=axL.transAxes, clip_on=False))
    axL.add_patch(Rectangle((frac, 0.28), 1 - frac, 0.2, color="#cfd8e0",
                            transform=axL.transAxes, clip_on=False))
    axL.text(0, 0.19, f"{a0:.1f} alternate", ha="left", color=LINEAGE_COLORS[L0], weight="bold",
             fontsize=11, transform=axL.transAxes)
    axL.text(1.0, 0.19, f"{b0:.1f} reference", ha="right", color="#1e2d3d", weight="bold",
             fontsize=11, transform=axL.transAxes)
    axL.text(0, 0.05, rf"${a0:.1f}/{kap[0]:.0f} = {mu[0]:.2f}$; total support sets the initial weight.",
             fontsize=10, color="#5b6b7b", transform=axL.transAxes)

    # Bottom-right: joint composition
    axR = fig.add_subplot(gs[1, 2:])
    axR.axis("off")
    axR.text(0, 0.92, "ONE JOINTLY ESTIMATED COMPOSITION", fontsize=11, weight="bold", color="#5b6b7b",
             transform=axR.transAxes)
    axR.text(1.0, 0.92, r"$\hat\theta_k \geq 0$,   $\sum_k \hat\theta_k = 1$", ha="right", fontsize=11,
             transform=axR.transAxes)
    x = 0.0
    for L in lineages:
        w = float(theta[L])
        axR.add_patch(Rectangle((x, 0.35), w, 0.3, color=LINEAGE_COLORS.get(L, "#333"),
                                transform=axR.transAxes, clip_on=False))
        if w > 0.06:
            axR.text(x + w / 2, 0.5, f"{100 * w:.1f}%", ha="center", va="center", color="white",
                     weight="bold", fontsize=12, transform=axR.transAxes)
        x += w
    n_l = len(lineages)
    for j, L in enumerate(lineages):
        axR.text((j + 0.5) / n_l, 0.14, f"{LINEAGE_LABELS.get(L, L)}  {100 * float(theta[L]):.1f}%",
                 ha="center", color=LINEAGE_COLORS.get(L, "#333"), weight="bold", fontsize=11,
                 transform=axR.transAxes)
    axR.text(0, -0.1, "All contributing mutations enter the same constrained fit.", fontsize=10,
             color="#5b6b7b", transform=axR.transAxes)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> str:
    tables = load_tables(LIKELIHOOD_TABLES, PRIORS_DIR)
    site, date = resolve_site_date(tables["theta"], tables["res"], SITE, DATE)
    theta_df, S, r = build_matrices(tables, site, date)
    theta = theta_df
    validate(theta, S, tables["priors"])
    rows = select_mutations(S, r, tables["lev"], tables["priors"], site, date,
                            SELECTION, N_PER_LINEAGE, MANUAL_MUTATIONS)
    if not rows:
        raise ValueError("No mutations selected")
    safe_site = "".join(ch if ch.isalnum() else "_" for ch in site).strip("_")
    out_path = os.path.join(OUT_DIR, f"stage2_evidence_figure_{safe_site}_{date}_{SELECTION}.png")
    plot_figure(theta, S, r, tables["priors"], rows, site, date, out_path)
    return out_path


if __name__ == "__main__":
    print(f"[FIGURE] {main()}")
