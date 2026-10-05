"""Decoding-graph topology and the groups used for partial pooling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymatching
import stim

BOUNDARY = -1


@dataclass
class Topology:
    """Which edges exist and which logical observable each one flips.

    Only the structure is taken from the reference model. Its probabilities
    are kept in `ref_p` for the vendor baseline but the estimators never read them.
    """

    n_detectors: int
    a: np.ndarray  # (edges,) first detector
    b: np.ndarray  # (edges,) second detector, or BOUNDARY
    fault_ids: list[set]
    ref_p: np.ndarray
    time_group: np.ndarray  # same physical edge repeated across bulk rounds
    class_group: np.ndarray  # same geometric kind of edge anywhere on the patch

    @property
    def is_boundary(self) -> np.ndarray:
        return self.b == BOUNDARY


def topology_from_dem(dem: stim.DetectorErrorModel, coords: np.ndarray) -> Topology:
    matching = pymatching.Matching.from_detector_error_model(dem)
    a, b, faults, ref_p = [], [], [], []
    for n1, n2, attrs in matching.edges():
        if n2 is not None and n1 > n2:
            n1, n2 = n2, n1
        a.append(n1)
        b.append(BOUNDARY if n2 is None else n2)
        faults.append(set(attrs["fault_ids"]))
        ref_p.append(attrs["error_probability"])
    a, b = np.array(a), np.array(b)
    time_group, class_group = _pooling_groups(a, b, faults, coords)
    return Topology(len(coords), a, b, faults, np.array(ref_p), time_group, class_group)


def _pooling_groups(a, b, faults, coords):
    """Label every edge with two nested groups.

    time_group: the same pair of stabilizers with the same time offset, in a bulk
    round. The first and last layers are physically different (state preparation
    and data-qubit readout) so they are kept apart.
    class_group: same stabilizer type, same geometric offset, same layer kind.
    """
    t_max = coords[:, 2].max()
    basis_positions = {tuple(c[:2]) for c in coords if c[2] == 0}

    def layer(t_lo, t_hi):
        if t_lo == 0:
            return 0
        return 2 if t_hi == t_max else 1

    time_keys, class_keys = [], []
    for i, j, f in zip(a, b, faults):
        ci = coords[i]
        kind_i = tuple(ci[:2]) in basis_positions
        if j == BOUNDARY:
            lay = layer(ci[2], ci[2])
            time_keys.append(("B", tuple(ci[:2]), lay, bool(f)))
            class_keys.append(("B", kind_i, lay, bool(f)))
            continue
        cj = coords[j]
        lay = layer(min(ci[2], cj[2]), max(ci[2], cj[2]))
        offset = tuple(cj - ci)
        time_keys.append(("E", tuple(ci[:2]), tuple(cj[:2]), offset[2], lay))
        # mirror-symmetric offsets belong to the same geometric class
        class_keys.append(("E", kind_i, tuple(cj[:2]) in basis_positions, abs(offset[0]), abs(offset[1]), offset[2], lay))
    return _encode(time_keys), _encode(class_keys)


def _encode(keys):
    table = {}
    return np.array([table.setdefault(k, len(table)) for k in keys])


def build_matching(topology: Topology, p: np.ndarray) -> pymatching.Matching:
    matching = pymatching.Matching()
    weights = np.log((1 - p) / p)
    for i, j, f, pe, w in zip(topology.a, topology.b, topology.fault_ids, p, weights):
        if j == BOUNDARY:
            matching.add_boundary_edge(int(i), fault_ids=f, weight=float(w), error_probability=float(pe))
        else:
            matching.add_edge(int(i), int(j), fault_ids=f, weight=float(w), error_probability=float(pe))
    matching.ensure_num_fault_ids(1)
    return matching
