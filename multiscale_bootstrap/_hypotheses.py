"""Extract edge, arc, and graph hypotheses from bootstrap results."""

import hashlib
import warnings
from collections import Counter

import numpy as np
import pandas as pd

# Edge-state values
STATE_NONE = 0
STATE_DIRECTED = 1
STATE_BIDIRECTED = 2


def _validate_min_causal_effect(min_causal_effect):
    """Validate a causal-effect threshold.

    Parameters
    ----------
    min_causal_effect : float or None
        Minimum absolute coefficient used to classify directed edges.

    Raises
    ------
    ValueError
        If the threshold is negative.
    """
    if min_causal_effect is not None and min_causal_effect < 0:
        raise ValueError("min_causal_effect must be non-negative or None.")


def encode_edge_states(adjacency, min_causal_effect):
    """Encode adjacency matrices into 3-valued edge-state matrices.

    Parameters
    ----------
    adjacency : ndarray, shape (..., d, d)
        One or more adjacency matrices with the ``adjacency[to, from]``
        convention used by ``lingam`` models. ``NaN`` marks a bidirected
        edge.
    min_causal_effect : float or None
        Minimum absolute finite coefficient for a directed edge. ``None``
        treats every finite nonzero coefficient as directed.

    Returns
    -------
    state : ndarray of int8, shape (..., d, d)
        ``2`` = bidirected, ``1`` = directed ``from -> to``, ``0`` = none.
    """
    _validate_min_causal_effect(min_causal_effect)
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim < 2 or adjacency.shape[-2] != adjacency.shape[-1]:
        raise ValueError("adjacency must have shape (..., d, d).")

    nan_mask = np.isnan(adjacency)
    symmetric_nan_mask = nan_mask | np.swapaxes(nan_mask, -1, -2)
    if not np.array_equal(nan_mask, symmetric_nan_mask):
        warnings.warn(
            "Bidirected (NaN) markers in the adjacency matrix were not "
            "symmetric; symmetrizing before encoding edge states.",
            stacklevel=2,
        )

    finite_mask = np.isfinite(adjacency)
    if min_causal_effect is None:
        directed_mask = finite_mask & (adjacency != 0)
    else:
        directed_mask = finite_mask & (
            np.abs(adjacency) > min_causal_effect
        )

    state = np.zeros(adjacency.shape, dtype=np.int8)
    state[directed_mask] = STATE_DIRECTED
    state[symmetric_nan_mask] = STATE_BIDIRECTED
    diagonal = np.arange(adjacency.shape[-1])
    state[..., diagonal, diagonal] = STATE_NONE
    return state


def _encode_scale_states(adjacency_matrices, min_causal_effect):
    """Encode all successful adjacency matrices for one bootstrap scale.

    Parameters
    ----------
    adjacency_matrices : ndarray, shape (n_success, d, d)
        Raw adjacency matrices for one scale.
    min_causal_effect : float or None
        Minimum absolute coefficient for a directed edge.

    Returns
    -------
    ndarray, shape (n_success, d, d)
        Three-valued edge-state matrices.
    """
    _validate_min_causal_effect(min_causal_effect)
    adjacency_matrices = np.asarray(adjacency_matrices, dtype=float)
    if (
        adjacency_matrices.ndim != 3
        or adjacency_matrices.shape[1] != adjacency_matrices.shape[2]
    ):
        raise ValueError(
            "adjacency_matrices must have shape (n_success, d, d)."
        )
    return encode_edge_states(adjacency_matrices, min_causal_effect)


def edge_hypothesis_id(from_index, to_index):
    """Return the canonical identifier for a directed-edge hypothesis.

    Parameters
    ----------
    from_index : int
        Source variable index.
    to_index : int
        Destination variable index.

    Returns
    -------
    str
        Identifier in the form ``"edge:from->to"``.
    """
    return f"edge:{from_index}->{to_index}"


def arc_hypothesis_id(i, j):
    """Return the canonical identifier for a bidirected-arc hypothesis.

    Parameters
    ----------
    i, j : int
        Variable indices. Their order does not affect the identifier.

    Returns
    -------
    str
        Identifier in the form ``"arc:lower<->higher"``.
    """
    lo, hi = (i, j) if i < j else (j, i)
    return f"arc:{lo}<->{hi}"


def graph_hypothesis_id(full_hash_hex):
    """Return the identifier for a full-graph hypothesis.

    Parameters
    ----------
    full_hash_hex : str
        SHA-1 digest of a graph-state matrix.

    Returns
    -------
    str
        Identifier prefixed with ``"graph:"``.
    """
    return f"graph:{full_hash_hex}"


def compute_graph_hash(state):
    """Compute a deterministic SHA-1 digest for a graph-state matrix.

    Parameters
    ----------
    state : ndarray of int8, shape (d, d)
        Three-valued graph-state matrix.

    Returns
    -------
    str
        Full SHA-1 hexadecimal digest. The matrix shape is included in the
        digest payload.
    """
    payload = state.astype(np.int8).tobytes() + str(state.shape).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


_BP_COLUMNS = [
    "hypothesis_id",
    "hypothesis_type",
    "scale_index",
    "scale_ratio",
    "n",
    "n_prime",
    "sa",
    "bp_count",
    "nb",
    "bp",
    "is_valid_scale",
    "from_index",
    "to_index",
    "graph_hash",
]


def _bp_row(
    result,
    scale,
    hypothesis_id,
    hypothesis_type,
    bp_count,
    bp,
    from_index=None,
    to_index=None,
    graph_hash=None,
):
    """Build one row for the common bootstrap-probability schema.

    Parameters
    ----------
    result : MultiScaleBootstrapResult
        Aggregate bootstrap result.
    scale : SingleScaleBootstrapResult
        Scale associated with the row.
    hypothesis_id : str
        Canonical hypothesis identifier.
    hypothesis_type : str
        Hypothesis family name.
    bp_count : int
        Number of successful replicates supporting the hypothesis.
    bp : float
        Bootstrap probability.
    from_index, to_index : int, optional
        Endpoint indices for pairwise hypotheses.
    graph_hash : str, optional
        Full graph hash for graph hypotheses.

    Returns
    -------
    dict
        Row matching the ``bp_long`` schema.
    """
    return dict(
        hypothesis_id=hypothesis_id,
        hypothesis_type=hypothesis_type,
        scale_index=scale.scale_index,
        scale_ratio=scale.scale_ratio,
        n=result.n,
        n_prime=scale.n_prime,
        sa=scale.sa,
        bp_count=bp_count,
        nb=scale.n_success,
        bp=bp,
        is_valid_scale=scale.is_valid,
        from_index=from_index,
        to_index=to_index,
        graph_hash=graph_hash,
    )


def _rows_to_bp_long(rows):
    """Normalize hypothesis rows to the common scaleboot boundary schema.

    Parameters
    ----------
    rows : sequence of dict
        Hypothesis rows containing the fields in ``_BP_COLUMNS``.

    Returns
    -------
    pandas.DataFrame
        Sorted long-format bootstrap-probability table.
    """
    frame = pd.DataFrame(rows, columns=_BP_COLUMNS)
    frame = frame.sort_values(
        ["hypothesis_type", "hypothesis_id", "scale_index"]
    ).reset_index(drop=True)
    frame["from_index"] = frame["from_index"].astype("Int64")
    frame["to_index"] = frame["to_index"].astype("Int64")
    frame["graph_hash"] = frame["graph_hash"].astype("string")
    return frame


class HypothesisResult:
    """Bootstrap probabilities for a hypothesis family.

    Parameters
    ----------
    bp_long : pandas.DataFrame
        Long-format bootstrap-probability table.

    Attributes
    ----------
    bp_long : pandas.DataFrame
        Bootstrap probabilities and scale metadata.
    """

    def __init__(self, bp_long):
        """Construct a hypothesis result.

        Parameters
        ----------
        bp_long : pandas.DataFrame
            Long-format bootstrap-probability table.
        """
        self.bp_long = bp_long


class GraphHypothesisResult(HypothesisResult):
    """Bootstrap probabilities and state matrices for graph hypotheses.

    Parameters
    ----------
    bp_long : pandas.DataFrame
        Long-format graph-hypothesis bootstrap-probability table.
    graph_states : dict of str to ndarray
        Representative state matrix for each graph hash.

    Attributes
    ----------
    graph_states : dict of str to ndarray
        Deduplicated graph-state matrices keyed by their SHA-1 digest.
    """

    def __init__(self, bp_long, graph_states):
        """Construct a graph-hypothesis result.

        Parameters
        ----------
        bp_long : pandas.DataFrame
            Long-format graph-hypothesis table.
        graph_states : dict of str to ndarray
            Deduplicated graph-state matrices keyed by graph hash.
        """
        super().__init__(bp_long)
        self.graph_states = graph_states


class EdgeHypothesis:
    """Extract directed-edge and bidirected-arc hypotheses together.

    The two relation kinds are one pairwise hypothesis family by
    specification, but their ``bp_long`` rows retain ``"edge"`` and
    ``"arc"`` as distinct ``hypothesis_type`` values.
    """

    hypothesis_type = "edge"

    def extract(self, result, min_causal_effect=None):
        """Extract directed-edge and bidirected-arc hypotheses.

        Parameters
        ----------
        result : MultiScaleBootstrapResult
            Bootstrap result containing raw adjacency matrices.
        min_causal_effect : float or None, optional
            Minimum absolute finite coefficient for a directed edge. ``None``
            treats every finite nonzero coefficient as directed.

        Returns
        -------
        HypothesisResult
            Pairwise hypotheses in ``bp_long``. Directed edges use
            ``hypothesis_type="edge"`` and bidirected relations use
            ``hypothesis_type="arc"``.
        """
        _validate_min_causal_effect(min_causal_effect)
        rows = []
        n_variables = len(result.variable_names)
        for scale in result.scales:
            states = _encode_scale_states(
                scale.adjacency_matrices, min_causal_effect
            )
            n_success = scale.n_success
            for source in range(n_variables):
                for destination in range(n_variables):
                    if source == destination:
                        continue
                    if n_success > 0:
                        count = int(
                            np.sum(states[:, destination, source] == STATE_DIRECTED)
                        )
                        bp = count / n_success
                    else:
                        count, bp = 0, float("nan")
                    rows.append(
                        _bp_row(
                            result,
                            scale,
                            edge_hypothesis_id(source, destination),
                            self.hypothesis_type,
                            count,
                            bp,
                            source,
                            destination,
                        )
                    )
            for lower in range(n_variables):
                for higher in range(lower + 1, n_variables):
                    if n_success > 0:
                        count = int(
                            np.sum(states[:, lower, higher] == STATE_BIDIRECTED)
                        )
                        bp = count / n_success
                    else:
                        count, bp = 0, float("nan")
                    rows.append(
                        _bp_row(
                            result,
                            scale,
                            arc_hypothesis_id(lower, higher),
                            "arc",
                            count,
                            bp,
                            lower,
                            higher,
                        )
                    )
        return HypothesisResult(bp_long=_rows_to_bp_long(rows))


class GraphHypothesis:
    """Extract whole-graph hypotheses independently from pairwise ones.

    A graph hypothesis is identified by the full SHA-1 hash of its encoded
    state matrix. Graphs absent from a scale receive zero bootstrap
    probability when that scale has successful replicates.
    """

    hypothesis_type = "graph"

    def extract(self, result, min_causal_effect=None):
        """Extract whole-graph hypotheses from a bootstrap result.

        Parameters
        ----------
        result : MultiScaleBootstrapResult
            Bootstrap result containing raw adjacency matrices.
        min_causal_effect : float or None, optional
            Minimum absolute finite coefficient for a directed edge. ``None``
            treats every finite nonzero coefficient as directed.

        Returns
        -------
        GraphHypothesisResult
            Graph-hypothesis probabilities and deduplicated state matrices.
        """
        _validate_min_causal_effect(min_causal_effect)
        rows = []
        graph_states = {}
        graph_seen = {}

        for scale in result.scales:
            states = _encode_scale_states(
                scale.adjacency_matrices, min_causal_effect
            )
            if scale.n_success == 0:
                continue
            graph_hashes = []
            for state in states:
                full_hash = compute_graph_hash(state)
                graph_hashes.append(full_hash)
                if full_hash not in graph_states:
                    graph_states[full_hash] = state.copy()
            counts = Counter(graph_hashes)
            for full_hash, count in counts.items():
                hypothesis_id = graph_hypothesis_id(full_hash)
                graph_seen.setdefault(hypothesis_id, set()).add(scale.scale_index)
                rows.append(
                    _bp_row(
                        result,
                        scale,
                        hypothesis_id,
                        self.hypothesis_type,
                        count,
                        count / scale.n_success,
                        graph_hash=full_hash,
                    )
                )

        for hypothesis_id, seen_scales in graph_seen.items():
            full_hash = hypothesis_id[len("graph:") :]
            for scale in result.scales:
                if scale.scale_index in seen_scales:
                    continue
                count = 0
                bp = 0.0 if scale.n_success > 0 else float("nan")
                rows.append(
                    _bp_row(
                        result,
                        scale,
                        hypothesis_id,
                        self.hypothesis_type,
                        count,
                        bp,
                        graph_hash=full_hash,
                    )
                )

        return GraphHypothesisResult(
            bp_long=_rows_to_bp_long(rows),
            graph_states=graph_states,
        )
