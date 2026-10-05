"""Load Google's public surface-code experiments (Zenodo 6804040)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

GOOGLE_DECODERS = ("pymatching", "correlated_matching", "belief_matching", "tensor_network_contraction")
_NAME = re.compile(r"surface_code_b(?P<basis>[XZ])_d(?P<d>\d+)_r(?P<r>\d+)_center_(?P<row>\d+)_(?P<col>\d+)")


@dataclass
class Experiment:
    name: str
    basis: str
    distance: int
    rounds: int
    patch: str
    detections: np.ndarray  # (shots, detectors) bool, in the order the shots were taken
    obs: np.ndarray  # (shots,) uint8, actual logical flips
    coords: np.ndarray  # (detectors, 3) x, y, t
    path: Path

    def vendor_dem(self) -> stim.DetectorErrorModel:
        """Google's Pauli-model prior, built from component benchmarks."""
        return stim.DetectorErrorModel.from_file(str(self.path / "circuit_detector_error_model.dem"))

    def google_pij_dem(self, trained_on: str) -> stim.DetectorErrorModel:
        """Google's own data-driven prior. trained_on is 'even' or 'odd' (25k shots)."""
        other = "odd" if trained_on == "even" else "even"
        return stim.DetectorErrorModel.from_file(str(self.path / f"pij_from_{trained_on}_for_{other}.dem"))

    def google_prediction(self, decoder: str) -> np.ndarray:
        return _read_01(self.path / f"obs_flips_predicted_by_{decoder}.01")


def _read_01(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes().replace(b"\n", b""), dtype=np.uint8) - ord("0")


def list_experiments(root: str | Path, distance: int | None = None) -> list[Path]:
    paths = []
    for p in sorted(Path(root).glob("surface_code_*")):
        m = _NAME.fullmatch(p.name)
        if m and (distance is None or int(m["d"]) == distance):
            paths.append(p)
    return paths


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path)
    m = _NAME.fullmatch(path.name)
    if not m:
        raise ValueError(f"not a surface code experiment folder: {path}")
    circuit = stim.Circuit.from_file(str(path / "circuit_ideal.stim"))
    n_det = circuit.num_detectors
    detections = stim.read_shot_data_file(
        path=str(path / "detection_events.b8"), format="b8", num_detectors=n_det, num_observables=0
    )
    coord_map = circuit.get_detector_coordinates()
    coords = np.array([coord_map[i][:3] for i in range(n_det)], dtype=float)
    return Experiment(
        name=path.name,
        basis=m["basis"],
        distance=int(m["d"]),
        rounds=int(m["r"]),
        patch=f"{m['row']}_{m['col']}",
        detections=detections,
        obs=_read_01(path / "obs_flips_actual.01"),
        coords=coords,
        path=path,
    )
