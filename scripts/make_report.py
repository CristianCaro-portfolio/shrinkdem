"""Turn the CSVs in results/ into summary tables (results/summary.md) and one figure."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shrinkdem import fitted_error_per_cycle  # noqa: E402

RESULTS = Path("results")
LABELS = {
    "uniform": "Uniform prior (no knowledge)",
    "vendor_prior": "Vendor prior (Google benchmarks)",
    "google_pij_25k": "Google p_ij model, 25,000 shots",
    "raw_pij": "Raw p_ij",
    "raw_pij_floor1e-3": "Raw p_ij, floor 0.001",
    "shrunk_v1": "shrinkdem v1 (pooling)",
    "shrunk_v2": "shrinkdem v2 (pooling + budgets)",
    "shrunk_v2_vendor_ref": "shrinkdem v2 around vendor prior",
    "google_correlated_matching": "Google correlated matching (stronger decoder)",
    "google_belief_matching": "Google belief matching (stronger decoder)",
    "google_tensor_network_contraction": "Google tensor network (stronger decoder)",
    "google_pymatching": "Google's own PyMatching run",
}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"
SERIES = {"raw_pij": "#eb6834", "shrunk_v1": "#1baf7a", "shrunk_v2": "#2a78d6"}


def per_cycle(group: pd.DataFrame) -> float:
    """Fit one error-per-cycle per (basis, patch) curve, then average the curves."""
    fits = []
    for _, curve in group.groupby(["basis", "patch"]):
        curve = curve.sort_values("rounds")
        fits.append(fitted_error_per_cycle(curve["rounds"], curve["errors"] / curve["test_shots"]))
    return float(np.mean(fits))


def benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (distance, method, shots), g in df.groupby(["distance", "method", "train_shots"]):
        worse, better = g["worse_than_vendor"].sum(), g["better_than_vendor"].sum()
        rows.append({
            "distance": distance, "method": method, "train_shots": shots,
            "logical_errors": int(g["errors"].sum()), "test_shots": int(g["test_shots"].sum()),
            "error_per_cycle_pct": 100 * per_cycle(g),
            "paired_z_vs_vendor": (worse - better) / np.sqrt(worse + better) if worse + better else 0.0,
        })
    table = pd.DataFrame(rows)
    vendor = table[table.method == "vendor_prior"].set_index("distance")["error_per_cycle_pct"]
    table["vs_vendor_pct"] = 100 * (table["error_per_cycle_pct"] / table["distance"].map(vendor) - 1)
    return table


def figure(table: pd.DataFrame, path: Path) -> None:
    distances = sorted(table.distance.unique())
    fig, axes = plt.subplots(1, len(distances), figsize=(5.5 * len(distances), 4.6), facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    for ax, distance in zip(axes, distances):
        t = table[table.distance == distance]
        ax.set_facecolor(SURFACE)
        for method, color in SERIES.items():
            line = t[t.method == method].sort_values("train_shots")
            ax.plot(line["train_shots"], line["error_per_cycle_pct"], color=color, lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[method])
        vendor = t[t.method == "vendor_prior"]["error_per_cycle_pct"].iloc[0]
        ax.axhline(vendor, color=MUTED, lw=1.2, ls=(0, (4, 3)), label="Vendor prior (device benchmarks)")
        ax.set_xscale("log")
        ax.set_xticks([250, 1000, 4000, 16000, 30000], ["250", "1,000", "4,000", "16,000", "30,000"])
        ax.minorticks_off()
        ax.set_title(f"Distance {distance}", loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_xlabel("Shots used to learn the prior", color=MUTED, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("Logical error per cycle (%), lower is better", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.suptitle("Logical error left by each prior on 20,000 held-out shots per run (Google Sycamore, 2022 data)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def drift_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(drifting=np.where(df["drift_z"].abs() >= 5, "drifting runs (|z| >= 5)", "stable runs"))
    cols = ["vendor", "static", "trailing", "expanding"]
    out = df.groupby(["distance", "drifting"]).agg(runs=("experiment", "nunique"), shots=("block_shots", "sum"),
                                                    **{c: (c, "sum") for c in cols}).reset_index()
    total = df.groupby("distance").agg(runs=("experiment", "nunique"), shots=("block_shots", "sum"),
                                       **{c: (c, "sum") for c in cols}).reset_index().assign(drifting="all runs")
    out = pd.concat([out, total]).sort_values(["distance", "drifting"])
    for c in cols[1:]:
        out[f"{c}_vs_vendor_pct"] = 100 * (out[c] / out["vendor"] - 1)
    return out


def main() -> None:
    lines = ["# Results", ""]
    bench = pd.read_csv(RESULTS / "benchmark.csv")
    table = benchmark_table(bench)
    table.to_csv(RESULTS / "benchmark_summary.csv", index=False)
    figure(table, RESULTS / "sample_efficiency.png")
    lines += ["## Sample efficiency", "",
              f"{bench.experiment.nunique()} runs, {int(bench[bench.method == 'vendor_prior'].test_shots.sum()):,} held-out shots. "
              "Error per cycle is fitted over rounds per patch and basis, then averaged. "
              "paired z > 0 means more logical errors than the vendor prior on the same shots.", ""]
    for distance in sorted(table.distance.unique()):
        t = table[table.distance == distance].copy()
        t["method"] = t["method"].map(LABELS)
        t = t.sort_values(["train_shots", "error_per_cycle_pct"])
        lines += [f"### Distance {distance}", "", "| prior | shots used | logical errors | error per cycle % | vs vendor % | paired z |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r.method} | {r.train_shots:,} | {r.logical_errors:,} | {r.error_per_cycle_pct:.3f} | "
                  f"{r.vs_vendor_pct:+.1f} | {r.paired_z_vs_vendor:+.1f} |" for r in t.itertuples()]
        lines.append("")
    drift_files = sorted(RESULTS.glob("drift_d*.csv"))
    if drift_files:
        drift = drift_table(pd.concat(pd.read_csv(f) for f in drift_files))
        drift.to_csv(RESULTS / "drift_summary.csv", index=False)
        lines += ["## Stale prior versus fresh prior", "",
                  "Logical errors when each run is replayed in order (blocks of 2,000 shots after the first 4,000).", "",
                  "| distance | runs | n runs | shots | vendor | static 4k | trailing 4k | expanding | trailing vs vendor % | expanding vs vendor % |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r.distance} | {r.drifting} | {r.runs} | {int(r.shots):,} | {r.vendor:,} | {r.static:,} | {r.trailing:,} | "
                  f"{r.expanding:,} | {r.trailing_vs_vendor_pct:+.2f} | {r.expanding_vs_vendor_pct:+.2f} |" for r in drift.itertuples()]
        lines.append("")
    bursts = RESULTS / "bursts.json"
    if bursts.exists():
        b = json.loads(bursts.read_text())
        lines += ["## Burst monitor (repetition code, 500,000 shots)", "",
                  f"Flagged {b['shots_flagged']} shots in {len(b['bursts'])} bursts. They contain {b['logical_errors_flagged']} of the "
                  f"{b['logical_errors_total']} logical errors of the whole run. Logical error per shot: "
                  f"{b['error_rate_all']:.1e} with every shot, {b['error_rate_unflagged']:.1e} without the flagged ones.", ""]
        lines += ["| first shot | last shot | peak z | logical errors inside |", "|---|---|---|---|"]
        lines += [f"| {x['first_shot']} | {x['last_shot']} | {x['peak_z']} | {x['logical_errors_inside']} |" for x in b["bursts"]]
        lines.append("")
    (RESULTS / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
