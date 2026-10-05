"""Streaming health checks that need nothing but the detection events."""
from __future__ import annotations

import numpy as np


def detection_fraction(detections: np.ndarray, block: int) -> np.ndarray:
    """Share of detectors that fired, averaged over consecutive blocks of shots."""
    n = len(detections) // block * block
    return detections[:n].reshape(n // block, block, -1).mean(axis=(1, 2))


def drift_score(detections: np.ndarray, head: int = 10_000, tail: int = 10_000) -> float:
    """z-score of the change in detection fraction between the start and the end of a run."""
    a = detections[:head].mean(axis=1)
    b = detections[-tail:].mean(axis=1)
    return float((b.mean() - a.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)))


def find_bursts(detections: np.ndarray, window: int = 20, threshold: float = 8.0) -> list[tuple[int, int, float]]:
    """Flag runs of shots whose detection count jumps far above the recent norm.

    Uses a robust baseline (median and MAD of per-window means), so a burst cannot
    hide by inflating its own baseline. Returns (first_shot, last_shot, peak_z).
    """
    counts = detections.sum(axis=1).astype(float)
    n = len(counts) // window * window
    means = counts[:n].reshape(-1, window).mean(axis=1)
    center = np.median(means)
    spread = 1.4826 * np.median(np.abs(means - center)) or 1.0
    z = (means - center) / spread
    hot = np.flatnonzero(z > threshold)
    bursts = []
    for group in np.split(hot, np.flatnonzero(np.diff(hot) > 1) + 1) if len(hot) else []:
        bursts.append((int(group[0] * window), int((group[-1] + 1) * window - 1), float(z[group].max())))
    return bursts
