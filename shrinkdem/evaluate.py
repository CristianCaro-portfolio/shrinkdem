"""Decode held-out shots and score a prior by the logical errors it leaves."""
from __future__ import annotations

import numpy as np

from .graph import Topology, build_matching


def decode(topology: Topology, p: np.ndarray, detections: np.ndarray) -> np.ndarray:
    return build_matching(topology, p).decode_batch(detections)[:, 0]


def error_per_cycle(p_logical: float, rounds: int) -> float:
    """Logical error per round implied by the error probability after `rounds` rounds."""
    fidelity = max(1 - 2 * p_logical, 1e-9)
    return 0.5 * (1 - fidelity ** (1 / rounds))


def fitted_error_per_cycle(rounds: np.ndarray, p_logical: np.ndarray, min_rounds: int = 3) -> float:
    """Slope of log fidelity versus rounds, as in the Google papers."""
    rounds, p_logical = np.asarray(rounds, float), np.asarray(p_logical, float)
    use = rounds >= min_rounds
    if use.sum() < 2:
        return float("nan")  # a partial run without enough round counts to fit a slope
    slope = np.polyfit(rounds[use], np.log(np.clip(1 - 2 * p_logical[use], 1e-9, None)), 1)[0]
    return 0.5 * (1 - np.exp(slope))
