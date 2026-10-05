"""Edge probabilities for a matching decoder, learned from detection events alone.

`raw_pij` is the classic correlation estimator (Spitz et al. 2018, used by Google
since 2021). `shrunk_pij` adds two-level empirical-Bayes partial pooling so the
estimate stays usable when only a few hundred shots are available.
"""
from __future__ import annotations

import numpy as np

from .graph import Topology

P_FLOOR = 1e-4
P_CEIL = 0.4


def moments(detections: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean of every detector and the full matrix of pair coincidences."""
    x = np.ascontiguousarray(detections, dtype=np.float32)
    n = x.shape[0]
    return x.mean(axis=0).astype(np.float64), (x.T @ x).astype(np.float64) / n, n


def _pair_estimates(topology: Topology, mean, second, n):
    pair = ~topology.is_boundary
    i, j = topology.a[pair], topology.b[pair]
    mi, mj, mij = mean[i], mean[j], second[i, j]
    cov = mij - mi * mj
    denom = 1 - 2 * (mi + mj - 2 * mij)
    p = 0.5 - 0.5 * np.sqrt(np.clip(1 - 4 * cov / denom, 0, None))
    # sampling variance of a covariance between two binary variables, then the delta method
    pos = np.clip(cov, 0, None)
    var_cov = (mi * (1 - mi) * mj * (1 - mj) + pos * (1 - 2 * mi) * (1 - 2 * mj) - pos**2) / n
    var = var_cov / (denom**2 * np.clip(1 - 4 * pos / denom, 0.05, None))
    return pair, p, var


def _boundary_estimates(topology: Topology, mean, n, pair, p_pair, var_pair):
    """What is left of each detector's firing rate once its pair edges are explained."""
    i, j = topology.a[pair], topology.b[pair]
    log_term = np.log(np.clip(1 - 2 * np.clip(p_pair, 0, P_CEIL), 1e-6, None))
    log_prod = np.zeros(topology.n_detectors)
    np.add.at(log_prod, i, log_term)
    np.add.at(log_prod, j, log_term)
    var_sum = np.zeros(topology.n_detectors)
    np.add.at(var_sum, i, var_pair)
    np.add.at(var_sum, j, var_pair)
    k = topology.a[topology.is_boundary]
    prod = np.exp(log_prod[k])
    p = 0.5 - (1 - 2 * mean[k]) / (2 * prod)
    var = mean[k] * (1 - mean[k]) / (n * prod**2) + var_sum[k]
    return p, var


def _group_shrink(est, var, groups):
    """Shrink noisy estimates toward their group mean (James-Stein / empirical Bayes).

    Returns the shrunk values, their variance, and per-element group mean,
    variance of that mean, and shrinkage weight.
    """
    out = np.empty_like(est)
    out_var = np.empty_like(est)
    mu_e = np.empty_like(est)
    mu_var_e = np.empty_like(est)
    keep = np.empty_like(est)
    order = np.argsort(groups, kind="stable")
    bounds = np.flatnonzero(np.diff(groups[order])) + 1
    for idx in np.split(order, bounds):
        e, v = est[idx], var[idx]
        mu = e.mean()
        noise = v.mean()
        tau2 = max(e.var(ddof=1) - noise, 0.0) if len(idx) > 1 else 0.0
        b = tau2 / (tau2 + v) if len(idx) > 1 else np.ones_like(v)
        mu_var = (tau2 + noise) / len(idx)
        out[idx] = mu + b * (e - mu)
        out_var[idx] = b * v + (1 - b) ** 2 * mu_var
        mu_e[idx], mu_var_e[idx], keep[idx] = mu, mu_var, b
    return out, out_var, mu_e, mu_var_e, keep


def _two_level(est, var, time_group, class_group):
    """Pool each edge across rounds, then pool those means across the patch."""
    _, _, mu, mu_var, keep = _group_shrink(est, var, time_group)
    first = np.unique(time_group, return_index=True)[1]
    mu_shrunk, mu_shrunk_var, *_ = _group_shrink(mu[first], mu_var[first], class_group[first])
    lookup = np.empty(time_group.max() + 1)
    lookup_var = np.empty(time_group.max() + 1)
    lookup[time_group[first]] = mu_shrunk
    lookup_var[time_group[first]] = mu_shrunk_var
    singleton = np.bincount(time_group)[time_group] == 1
    # an edge that has no copies in other rounds only gets the class-level pooling
    deviation = np.where(singleton, 0.0, keep * (est - mu))
    post_var = np.where(singleton, 0.0, keep * var) + lookup_var[time_group]
    return lookup[time_group] + deviation, post_var


def rake_pairs(topology: Topology, mean: np.ndarray, p_pair: np.ndarray, pair: np.ndarray, iterations: int = 200) -> np.ndarray:
    """Make pair edges add up to what each bulk detector actually fires.

    Events that light up three or more detectors raise the correlation of every
    pair they touch, so the pair estimates overshoot and leave too little for the
    boundary edges. A detector with no boundary edge gives a hard budget: its
    pair edges alone must explain its firing rate. We solve for one factor per
    detector (iterative proportional fitting) on those bulk detectors and give
    boundary detectors the average bulk factor.
    """
    i, j = topology.a[pair], topology.b[pair]
    bulk = np.ones(topology.n_detectors, bool)
    bulk[topology.a[topology.is_boundary]] = False
    if not bulk.any():
        return p_pair
    rate = -0.5 * np.log(1 - 2 * np.clip(p_pair, 0, P_CEIL))
    budget = -0.5 * np.log(1 - 2 * mean)
    scale = np.ones(topology.n_detectors)
    for _ in range(iterations):
        load = np.zeros(topology.n_detectors)
        np.add.at(load, i, rate * scale[j])
        np.add.at(load, j, rate * scale[i])
        # geometric damping: the plain symmetric update oscillates
        scale[bulk] = np.sqrt(scale[bulk] * budget[bulk] / np.maximum(load[bulk], 1e-12))
        scale[~bulk] = scale[bulk].mean()
    return 0.5 * (1 - np.exp(-2 * rate * scale[i] * scale[j]))


def raw_pij(topology: Topology, detections: np.ndarray, floor: float = P_FLOOR) -> np.ndarray:
    mean, second, n = moments(detections)
    pair, p_pair, var_pair = _pair_estimates(topology, mean, second, n)
    p_bound, _ = _boundary_estimates(topology, mean, n, pair, p_pair, var_pair)
    p = np.empty(len(topology.a))
    p[pair], p[~pair] = p_pair, p_bound
    return np.clip(np.nan_to_num(p, nan=floor), floor, P_CEIL)


def shrunk_pij(
    topology: Topology,
    detections: np.ndarray,
    floor: float = P_FLOOR,
    levels: str = "time+class",
    rake: bool = True,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Partial-pooling estimate of every edge probability.

    levels: 'time+class' (default), 'class' (ignore per-qubit detail) or 'time'.
    rake: enforce the bulk detector budgets (see rake_pairs).
    reference: optional prior for every edge (for example the vendor model). When
    given, what gets pooled is the disagreement between data and prior, so with
    few shots the answer falls back to the prior plus a class-wide correction.
    """
    mean, second, n = moments(detections)
    pair, p_pair, var_pair = _pair_estimates(topology, mean, second, n)
    tg, cg = _groups_for(levels, topology)
    ref = np.zeros(len(topology.a)) if reference is None else reference
    p_pair, var_pair = _two_level(p_pair - ref[pair], var_pair, tg[pair], cg[pair])
    p_pair = p_pair + ref[pair]
    if rake:
        p_pair = rake_pairs(topology, mean, p_pair, pair)
    p_bound, var_bound = _boundary_estimates(topology, mean, n, pair, p_pair, var_pair)
    p_bound, _ = _two_level(p_bound - ref[~pair], var_bound, tg[~pair], cg[~pair])
    p_bound = p_bound + ref[~pair]
    p = np.empty(len(topology.a))
    p[pair], p[~pair] = p_pair, p_bound
    return np.clip(np.nan_to_num(p, nan=floor), floor, P_CEIL)


def _groups_for(levels: str, topology: Topology):
    if levels == "time+class":
        return topology.time_group, topology.class_group
    if levels == "class":
        return topology.class_group, topology.class_group
    if levels == "time":
        return topology.time_group, topology.time_group
    raise ValueError(levels)


def uniform(topology: Topology, p: float = 0.01) -> np.ndarray:
    """Zero-knowledge prior: every edge equally likely."""
    return np.full(len(topology.a), p)
