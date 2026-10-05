"""Find high-energy bursts in the 500,000-shot repetition-code run using syndromes only."""
import json
import sys
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shrinkdem.monitor import find_bursts  # noqa: E402


def main(data="data/raw", out="results/bursts.json"):
    path = Path(data) / "repetition_code_bZ_d25_r50_center_5_5"
    n_det = stim.Circuit.from_file(str(path / "circuit_ideal.stim")).num_detectors
    det = stim.read_shot_data_file(path=str(path / "detection_events.b8"), format="b8", num_detectors=n_det,
                                   num_observables=0)
    actual = np.frombuffer((path / "obs_flips_actual.01").read_bytes().replace(b"\n", b""), np.uint8)
    predicted = np.frombuffer((path / "obs_flips_predicted_by_correlated_matching.01").read_bytes().replace(b"\n", b""), np.uint8)
    wrong = actual != predicted
    bursts = find_bursts(det)
    flagged = np.zeros(len(det), bool)
    report = []
    for first, last, z in bursts:
        flagged[first:last + 1] = True
        report.append({"first_shot": first, "last_shot": last, "peak_z": round(z, 1),
                       "logical_errors_inside": int(wrong[first:last + 1].sum()),
                       "shots": last - first + 1})
    summary = {
        "shots": int(len(det)), "logical_errors_total": int(wrong.sum()),
        "shots_flagged": int(flagged.sum()), "logical_errors_flagged": int(wrong[flagged].sum()),
        "error_rate_all": float(wrong.mean()), "error_rate_unflagged": float(wrong[~flagged].mean()),
        "bursts": report,
    }
    Path(out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
