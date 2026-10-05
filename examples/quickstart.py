"""Thirty-second demo on simulated data (no download needed).

Learns a decoder prior from 500 shots of syndromes and compares it with the
classic estimator and with the true error model.
"""
import sys
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shrinkdem as sd  # noqa: E402

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=5, rounds=10,
    after_clifford_depolarization=0.004, before_measure_flip_probability=0.01,
    after_reset_flip_probability=0.004, before_round_data_depolarization=0.004,
)
coord_map = circuit.get_detector_coordinates()
coords = np.array([coord_map[i][:3] for i in range(circuit.num_detectors)])
topology = sd.topology_from_dem(circuit.detector_error_model(decompose_errors=True), coords)

detections, obs = circuit.compile_detector_sampler(seed=1).sample(40_500, separate_observables=True)
train, test, truth = detections[:500], detections[500:], obs[500:, 0]

priors = {
    "true error model": topology.ref_p,
    "uniform": sd.uniform(topology),
    "raw p_ij, 500 shots": sd.raw_pij(topology, train),
    "shrinkdem, 500 shots": sd.shrunk_pij(topology, train),
}
for name, p in priors.items():
    errors = (sd.decode(topology, p, test) != truth).mean()
    print(f"{name:24s} logical error {100 * errors:.2f}%")
