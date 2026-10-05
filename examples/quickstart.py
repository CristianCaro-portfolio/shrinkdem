"""Quickstart on real hardware data (no download needed).

Uses one run from Google's Sycamore processor bundled in sample_data/. The prior is
learned from the first shots of the run, syndromes only, and scored on the last 20,000.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import shrinkdem as sd  # noqa: E402

exp = sd.load_experiment(ROOT / "sample_data" / "surface_code_bZ_d5_r09_center_5_5")
topology = sd.topology_from_dem(exp.vendor_dem(), exp.coords)
test, truth = exp.detections[30_000:], exp.obs[30_000:]

priors = {"vendor prior (Google benchmarks)": topology.ref_p, "uniform (no knowledge)": sd.uniform(topology)}
for shots in (500, 4000):
    train = exp.detections[:shots]
    priors[f"raw p_ij, {shots} shots"] = sd.raw_pij(topology, train)
    priors[f"shrinkdem, {shots} shots"] = sd.shrunk_pij(topology, train)

print(f"{exp.name}: {len(truth):,} held-out shots from the real chip")
for name, p in priors.items():
    errors = (sd.decode(topology, p, test) != truth).mean()
    print(f"{name:34s} logical error {100 * errors:.2f}%")
