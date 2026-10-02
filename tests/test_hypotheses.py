import numpy as np

from multiscale_bootstrap import (
    EdgeHypothesis,
    GraphHypothesis,
    MultiScaleBootstrapResult,
    SingleScaleBootstrapResult,
    encode_edge_states,
)

EDGE = np.array([[0.0, 0.0], [0.3, 0.0]])  # x0 -> x1
NONE = np.zeros((2, 2))
ARC = np.array([[0.0, np.nan], [np.nan, 0.0]])  # x0 <-> x1


def make_result():
    """Create two valid scales and one scale invalidated by a failed fit."""
    scale0 = SingleScaleBootstrapResult(
        0.7, 70, 100 / 70, 4, np.stack([EDGE, EDGE, NONE, ARC])
    )
    scale1 = SingleScaleBootstrapResult(
        1.0, 100, 1.0, 4, np.stack([EDGE, NONE, NONE, NONE])
    )
    invalid = SingleScaleBootstrapResult(
        1.4, 140, 100 / 140, 4, np.stack([EDGE]), error="RuntimeError: x"
    )
    return MultiScaleBootstrapResult([scale0, scale1, invalid], 100, ["x0", "x1"])


def bp_of(bp_long, hypothesis_id):
    rows = bp_long[bp_long["hypothesis_id"] == hypothesis_id]
    return dict(zip(rows["scale_index"], rows["bp"]))


def test_encode_edge_states():
    adjacency = np.array([[0.0, 0.005, np.nan], [0.3, 0.0, 0.0], [np.nan, -0.5, 0.0]])

    state = encode_edge_states(adjacency, min_causal_effect=0.01)
    assert state.tolist() == [[0, 0, 2], [1, 0, 0], [2, 1, 0]]

    # None treats every nonzero coefficient as a directed edge.
    state = encode_edge_states(adjacency)
    assert state[0, 1] == 1


def test_encode_edge_states_ignores_infinite_coefficients():
    adjacency = np.array([[0.0, np.inf], [-np.inf, 0.0]])
    assert encode_edge_states(adjacency).tolist() == [[0, 0], [0, 0]]


def test_encode_edge_states_accepts_stacked_matrices():
    state = encode_edge_states(np.stack([EDGE, ARC]))
    assert state.shape == (2, 2, 2)
    assert state[0, 1, 0] == 1
    assert state[1, 0, 1] == 2


def test_edge_hypothesis_bp_and_schema():
    bp_long = EdgeHypothesis().extract(make_result())

    assert list(bp_long.columns) == ["hypothesis_id", "scale_index", "sa", "nb", "bp"]
    assert set(bp_long["hypothesis_id"]) == {"edge:0->1", "edge:1->0", "arc:0<->1"}
    assert set(bp_long["scale_index"]) == {0, 1}
    assert bp_of(bp_long, "edge:0->1") == {0: 0.5, 1: 0.25}
    assert bp_of(bp_long, "edge:1->0") == {0: 0.0, 1: 0.0}
    assert bp_of(bp_long, "arc:0<->1") == {0: 0.25, 1: 0.0}
    assert (bp_long["nb"] == 4).all()


def test_edge_hypothesis_applies_min_causal_effect_at_extraction():
    result = make_result()
    assert bp_of(EdgeHypothesis().extract(result, 0.1), "edge:0->1")[0] == 0.5
    assert bp_of(EdgeHypothesis().extract(result, 0.5), "edge:0->1")[0] == 0.0


def test_graph_hypothesis_bp_and_states():
    bp_long, graph_states = GraphHypothesis().extract(make_result())

    assert list(bp_long.columns) == ["hypothesis_id", "scale_index", "sa", "nb", "bp"]
    assert len(graph_states) == 3
    assert all(i.startswith("graph:") for i in graph_states)

    edge_id = next(i for i, s in graph_states.items() if s[1, 0] == 1)
    none_id = next(i for i, s in graph_states.items() if not s.any())
    arc_id = next(i for i, s in graph_states.items() if s[0, 1] == 2)
    assert bp_of(bp_long, edge_id) == {0: 0.5, 1: 0.25}
    assert bp_of(bp_long, none_id) == {0: 0.25, 1: 0.75}
    # Graphs absent from a scale have zero BP at that scale.
    assert bp_of(bp_long, arc_id) == {0: 0.25, 1: 0.0}
    assert bp_long.groupby("scale_index")["bp"].sum().tolist() == [1.0, 1.0]


def test_no_valid_scale_gives_empty_bp_long():
    result = make_result()
    result.scales = [result.scales[2]]
    assert EdgeHypothesis().extract(result).empty
    bp_long, graph_states = GraphHypothesis().extract(result)
    assert bp_long.empty
    assert graph_states == {}
