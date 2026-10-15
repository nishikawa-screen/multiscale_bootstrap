"""Extract edge, arc, and graph hypotheses from bootstrap results.

Every extractor returns bootstrap probabilities in the long format
``bp_long`` with one row per (hypothesis, scale) and the columns
``hypothesis_id``, ``scale_index``, ``sa``, ``nb`` and ``bp``. Hypothesis
identifiers are ``"edge:from->to"``, ``"arc:lower<->higher"`` and
``"graph:<sha1>"``.
"""

import hashlib
from collections import Counter

import numpy as np
import pandas as pd

# Edge-state values
STATE_NONE = 0
STATE_DIRECTED = 1
STATE_BIDIRECTED = 2

_BP_COLUMNS = ["hypothesis_id", "scale_index", "sa", "nb", "bp"]


def encode_edge_states(adjacency, min_causal_effect=None):
    """Encode adjacency matrices into 3-valued edge-state matrices.

    Parameters
    ----------
    adjacency : ndarray, shape (..., d, d)
        One or more adjacency matrices with the ``adjacency[to, from]``
        convention used by ``lingam`` models. ``NaN`` marks a bidirected
        edge.
    min_causal_effect : float or None, optional
        Minimum absolute coefficient for a directed edge. ``None`` treats
        every nonzero coefficient as directed.

    Returns
    -------
    state : ndarray of int8, shape (..., d, d)
        ``2`` = bidirected, ``1`` = directed ``from -> to``, ``0`` = none.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    if min_causal_effect is None:
        min_causal_effect = 0.0

    state = np.zeros(adjacency.shape, dtype=np.int8)
    state[np.abs(np.nan_to_num(adjacency)) > min_causal_effect] = STATE_DIRECTED
    state[np.isnan(adjacency)] = STATE_BIDIRECTED
    return state


def _bp_rows(scale_index, scale, hypothesis_ids, bps):
    """Build ``bp_long`` rows for the hypotheses of one scale."""
    return [
        dict(
            hypothesis_id=hypothesis_id,
            scale_index=scale_index,
            sa=scale.sa,
            nb=scale.n_success,
            bp=bp,
        )
        for hypothesis_id, bp in zip(hypothesis_ids, bps)
    ]


class EdgeHypothesis:
    """Extract directed-edge and bidirected-arc hypotheses."""

    def extract(self, result, min_causal_effect=None):
        """Extract directed-edge and bidirected-arc hypotheses.

        Parameters
        ----------
        result : MultiScaleBootstrapResult
            Bootstrap result containing raw adjacency matrices.
        min_causal_effect : float or None, optional
            Minimum absolute coefficient for a directed edge. ``None``
            treats every nonzero coefficient as directed.

        Returns
        -------
        bp_long : pandas.DataFrame
            Bootstrap probabilities of the scales with at least one successful
            replicate (``nb`` is the success count). Directed edges have
            ``hypothesis_id="edge:from->to"`` and bidirected arcs
            ``"arc:lower<->higher"``.
        """
        d = len(result.variable_names)
        rows = []
        for scale_index, scale in enumerate(result.scales):
            if scale.n_success == 0:
                continue
            states = encode_edge_states(scale.adjacency_matrices_, min_causal_effect)
            directed = (states == STATE_DIRECTED).mean(axis=0)
            bidirected = (states == STATE_BIDIRECTED).mean(axis=0)

            ids, bps = [], []
            for to in range(d):
                for from_ in range(d):
                    if to != from_:
                        ids.append(f"edge:{from_}->{to}")
                        bps.append(directed[to, from_])
            for lower in range(d):
                for higher in range(lower + 1, d):
                    ids.append(f"arc:{lower}<->{higher}")
                    bps.append(bidirected[lower, higher])
            rows += _bp_rows(scale_index, scale, ids, bps)

        return pd.DataFrame(rows, columns=_BP_COLUMNS)


class GraphHypothesis:
    """Extract whole-graph hypotheses.

    A graph hypothesis is identified by the SHA-1 hash of its encoded state
    matrix. Graphs absent from a scale receive zero bootstrap probability.
    """

    def extract(self, result, min_causal_effect=None):
        """Extract whole-graph hypotheses from a bootstrap result.

        Parameters
        ----------
        result : MultiScaleBootstrapResult
            Bootstrap result containing raw adjacency matrices.
        min_causal_effect : float or None, optional
            Minimum absolute coefficient for a directed edge. ``None``
            treats every nonzero coefficient as directed.

        Returns
        -------
        bp_long : pandas.DataFrame
            Bootstrap probabilities of the scales with at least one successful
            replicate, with
            ``hypothesis_id="graph:<sha1>"``.
        graph_states : dict of str to ndarray
            State matrix of each graph, keyed by ``hypothesis_id``.
        """
        graph_states = {}
        counts = {}
        for scale_index, scale in enumerate(result.scales):
            if scale.n_success == 0:
                continue
            counter = Counter()
            for state in encode_edge_states(
                scale.adjacency_matrices_, min_causal_effect
            ):
                hypothesis_id = "graph:" + hashlib.sha1(state.tobytes()).hexdigest()
                graph_states.setdefault(hypothesis_id, state.copy())
                counter[hypothesis_id] += 1
            counts[scale_index] = counter

        rows = []
        for scale_index, counter in counts.items():
            scale = result.scales[scale_index]
            ids = list(graph_states)
            bps = [counter[i] / scale.n_success for i in ids]
            rows += _bp_rows(scale_index, scale, ids, bps)

        return pd.DataFrame(rows, columns=_BP_COLUMNS), graph_states
