"""Shared synthetic-data helpers for the multiscale_bootstrap test suite."""

import numpy as np

from multiscale_bootstrap import MultiScaleBootstrapResult, SingleScaleBootstrapResult


def make_chain_data(n=500, seed=0):
    """3-variable chain x0 -> x1 -> x2 with uniform (non-Gaussian) noise.

    Returns
    -------
    X : ndarray, shape (n, 3)
    """
    rng = np.random.default_rng(seed)
    e0 = rng.uniform(-1, 1, n)
    e1 = rng.uniform(-1, 1, n)
    e2 = rng.uniform(-1, 1, n)
    x0 = e0
    x1 = 0.8 * x0 + e1
    x2 = 0.6 * x1 + e2
    return np.column_stack([x0, x1, x2])


def make_bootstrap_result():
    """Return a small result containing edge, empty, and bidirected states."""
    edge_adjacency = np.array([[0.0, 0.0], [0.3, 0.0]])
    no_edge_adjacency = np.array([[0.0, 0.0], [0.0, 0.0]])
    arc_adjacency = np.array([[0.0, np.nan], [np.nan, 0.0]])

    scale0 = SingleScaleBootstrapResult(
        scale_index=0,
        scale_ratio=0.7,
        n_prime=70,
        sa=100 / 70,
        n_bootstrap_requested=4,
        n_success=4,
        adjacency_matrices=np.stack(
            [edge_adjacency, edge_adjacency, no_edge_adjacency, arc_adjacency]
        ),
        fit_failures=[],
        min_success_rate=0.5,
    )
    scale1 = SingleScaleBootstrapResult(
        scale_index=1,
        scale_ratio=1.4,
        n_prime=140,
        sa=100 / 140,
        n_bootstrap_requested=4,
        n_success=3,
        adjacency_matrices=np.stack(
            [edge_adjacency, no_edge_adjacency, no_edge_adjacency]
        ),
        fit_failures=[],
        min_success_rate=0.5,
    )

    return MultiScaleBootstrapResult(
        scales=[scale0, scale1],
        n=100,
        variable_names=["x0", "x1"],
    )
