"""Sample-efficiency benchmark on real hardware data.

For every experiment: learn the decoder prior from the first N shots (syndromes
only, no logical labels) and count logical errors on the last 20,000 shots.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shrinkdem as sd  # noqa: E402
from shrinkdem.io import GOOGLE_DECODERS  # noqa: E402

TEST_START = 30_000
SHOT_COUNTS = (250, 1000, 4000, 16000, 30000)
FIELDS = ["experiment", "distance", "basis", "rounds", "patch", "method", "train_shots", "errors", "test_shots",
          "worse_than_vendor", "better_than_vendor"]


def estimators(topology, train):
    yield "raw_pij", sd.raw_pij(topology, train)
    yield "raw_pij_floor1e-3", sd.raw_pij(topology, train, floor=1e-3)
    yield "shrunk_v1", sd.shrunk_pij(topology, train, rake=False)
    yield "shrunk_v2", sd.shrunk_pij(topology, train)
    yield "shrunk_v2_vendor_ref", sd.shrunk_pij(topology, train, reference=topology.ref_p)


def run_experiment(path, shot_counts):
    exp = sd.load_experiment(path)
    topology = sd.topology_from_dem(exp.vendor_dem(), exp.coords)
    test = exp.detections[TEST_START:]
    truth = exp.obs[TEST_START:]
    wrong_vendor = sd.decode(topology, topology.ref_p, test) != truth

    def row(method, shots, wrong):
        return {
            "experiment": exp.name, "distance": exp.distance, "basis": exp.basis, "rounds": exp.rounds,
            "patch": exp.patch, "method": method, "train_shots": shots, "errors": int(wrong.sum()),
            "test_shots": len(truth), "worse_than_vendor": int((wrong & ~wrong_vendor).sum()),
            "better_than_vendor": int((~wrong & wrong_vendor).sum()),
        }

    yield row("vendor_prior", 0, wrong_vendor)
    yield row("uniform", 0, sd.decode(topology, sd.uniform(topology), test) != truth)

    # Google's own data-driven prior: 25k alternating shots, scored only on the parity it did not see
    index = np.arange(TEST_START, len(exp.obs))
    wrong = np.zeros(len(truth), bool)
    for trained_on, held_out in (("even", 1), ("odd", 0)):
        google = sd.topology_from_dem(exp.google_pij_dem(trained_on), exp.coords)
        sel = index % 2 == held_out
        wrong[sel] = sd.decode(google, google.ref_p, test[sel]) != truth[sel]
    yield row("google_pij_25k", 25000, wrong)

    for decoder in GOOGLE_DECODERS:
        yield row(f"google_{decoder}", 0, exp.google_prediction(decoder)[TEST_START:] != truth)

    for shots in shot_counts:
        for method, p in estimators(topology, exp.detections[:shots]):
            yield row(method, shots, sd.decode(topology, p, test) != truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--out", default="results/benchmark.csv")
    parser.add_argument("--distance", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="only the first K experiments (smoke test)")
    args = parser.parse_args()

    paths = sd.list_experiments(args.data, args.distance)[: args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for k, path in enumerate(paths, 1):
            writer.writerows(run_experiment(path, SHOT_COUNTS))
            f.flush()
            print(f"[{k}/{len(paths)}] {path.name} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
