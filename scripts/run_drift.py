"""Does a fresh prior beat a stale one? Replay each run in the order it was taken.

Every 2,000-shot block is decoded with:
  vendor      Google's benchmark-based prior, fixed
  static      our prior learned once from the first 4,000 shots
  trailing    our prior relearned from the 4,000 shots just before the block
  expanding   our prior relearned from every shot before the block
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shrinkdem as sd  # noqa: E402
from shrinkdem.monitor import drift_score  # noqa: E402

WINDOW = 4_000
BLOCK = 2_000
FIELDS = ["experiment", "distance", "basis", "rounds", "patch", "drift_z", "block_start", "vendor", "static",
          "trailing", "expanding", "block_shots"]


def run_experiment(path):
    exp = sd.load_experiment(path)
    topology = sd.topology_from_dem(exp.vendor_dem(), exp.coords)
    x, y = exp.detections, exp.obs
    z = drift_score(x)
    static = sd.shrunk_pij(topology, x[:WINDOW])
    for start in range(WINDOW, len(y), BLOCK):
        block = slice(start, start + BLOCK)
        priors = {
            "vendor": topology.ref_p,
            "static": static,
            "trailing": sd.shrunk_pij(topology, x[start - WINDOW:start]),
            "expanding": sd.shrunk_pij(topology, x[:start]),
        }
        errors = {k: int((sd.decode(topology, p, x[block]) != y[block]).sum()) for k, p in priors.items()}
        yield {"experiment": exp.name, "distance": exp.distance, "basis": exp.basis, "rounds": exp.rounds,
               "patch": exp.patch, "drift_z": round(z, 2), "block_start": start, "block_shots": len(y[block]), **errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--out", default="results/drift.csv")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--min-rounds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    paths = [p for p in sd.list_experiments(args.data, args.distance) if int(p.name.split("_r")[1][:2]) >= args.min_rounds]
    paths = paths[: args.limit]
    start = time.time()
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for k, path in enumerate(paths, 1):
            writer.writerows(run_experiment(path))
            f.flush()
            print(f"[{k}/{len(paths)}] {path.name} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
