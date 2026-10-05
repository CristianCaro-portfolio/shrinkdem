"""Checks on simulated data, where the true error model is known."""
import numpy as np
import pytest
import stim

import shrinkdem as sd
from shrinkdem.monitor import find_bursts


@pytest.fixture(scope="module")
def simulated():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=5, rounds=8,
        after_clifford_depolarization=0.004, before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.004, before_round_data_depolarization=0.004,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    coord_map = circuit.get_detector_coordinates()
    coords = np.array([coord_map[i][:3] for i in range(circuit.num_detectors)])
    topology = sd.topology_from_dem(dem, coords)
    detections, obs = circuit.compile_detector_sampler(seed=7).sample(60_000, separate_observables=True)
    return topology, detections, obs[:, 0]


def test_recovers_known_probabilities(simulated):
    topology, detections, _ = simulated
    estimate = sd.shrunk_pij(topology, detections[:40_000])
    assert np.corrcoef(estimate, topology.ref_p)[0, 1] > 0.95
    assert abs(estimate.sum() / topology.ref_p.sum() - 1) < 0.1


def test_pooling_beats_raw_with_few_shots(simulated):
    topology, detections, _ = simulated
    truth = topology.ref_p
    raw = sd.raw_pij(topology, detections[:500])
    shrunk = sd.shrunk_pij(topology, detections[:500])
    pair = ~topology.is_boundary
    assert np.abs(shrunk - truth)[pair].mean() < 0.6 * np.abs(raw - truth)[pair].mean()
    assert np.abs(shrunk - truth)[~pair].mean() < np.abs(raw - truth)[~pair].mean()


def test_decoding_close_to_true_model(simulated):
    topology, detections, obs = simulated
    test = slice(40_000, 60_000)
    errors = lambda p: (sd.decode(topology, p, detections[test]) != obs[test]).mean()
    ideal = errors(topology.ref_p)
    assert errors(sd.shrunk_pij(topology, detections[:1000])) < ideal * 1.25 + 0.002
    assert errors(sd.shrunk_pij(topology, detections[:1000])) <= errors(sd.raw_pij(topology, detections[:1000]))


def test_monitor_finds_injected_burst():
    rng = np.random.default_rng(0)
    detections = rng.random((5000, 200)) < 0.1
    detections[3000:3100] |= rng.random((100, 200)) < 0.3
    bursts = find_bursts(detections)
    assert len(bursts) == 1 and 2980 <= bursts[0][0] <= 3020
