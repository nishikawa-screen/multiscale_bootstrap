import numpy as np
import pytest

import multiscale_bootstrap._hypotheses as hypotheses_module
from multiscale_bootstrap import (
    EdgeHypothesis,
    GraphHypothesis,
    SingleScaleBootstrapResult,
    compute_graph_hash,
    encode_edge_states,
)
from multiscale_bootstrap._hypotheses import (
    STATE_BIDIRECTED,
    STATE_DIRECTED,
    STATE_NONE,
)
from ._helpers import make_bootstrap_result


def test_encode_edge_states_without_threshold_uses_finite_nonzero_coefficients():
    adjacency = np.array([[0.0, np.inf], [0.3, 0.0]])
    state = encode_edge_states(adjacency, min_causal_effect=None)
    assert state[1, 0] == STATE_DIRECTED
    assert state[0, 1] == STATE_NONE


def test_encode_edge_states_zero_min_causal_effect():
    adjacency = np.array([[0.0, 0.0], [0.05, 0.0]])
    state = encode_edge_states(adjacency, min_causal_effect=0.0)
    assert state[1, 0] == STATE_DIRECTED


def test_encode_edge_states_positive_min_causal_effect_is_strict():
    adjacency = np.array([[0.0, 0.0], [0.05, 0.0]])
    state = encode_edge_states(adjacency, min_causal_effect=0.05)
    assert state[1, 0] == STATE_NONE


def test_encode_edge_states_rejects_negative_min_causal_effect():
    with pytest.raises(ValueError, match="min_causal_effect"):
        encode_edge_states(np.zeros((2, 2)), min_causal_effect=-0.01)


def test_encode_edge_states_bidirected_nan():
    adjacency = np.array([[0.0, np.nan], [np.nan, 0.0]])
    state = encode_edge_states(adjacency, min_causal_effect=None)
    assert state[0, 1] == STATE_BIDIRECTED
    assert state[1, 0] == STATE_BIDIRECTED


def test_encode_edge_states_warns_on_asymmetric_nan():
    adjacency = np.array([[0.0, np.nan], [0.0, 0.0]])
    with pytest.warns(UserWarning):
        state = encode_edge_states(adjacency, min_causal_effect=None)
    assert state[0, 1] == STATE_BIDIRECTED
    assert state[1, 0] == STATE_BIDIRECTED


def test_encode_edge_states_keeps_diagonal_empty():
    adjacency = np.array([[np.nan, 0.0], [0.0, 0.2]])
    state = encode_edge_states(adjacency, min_causal_effect=None)
    assert np.array_equal(np.diag(state), [STATE_NONE, STATE_NONE])


def test_encode_scale_states_preserves_shape_dtype_and_empty_input():
    adjacency_matrices = np.array(
        [
            [[0.0, 0.0], [0.2, 0.0]],
            [[0.0, np.nan], [np.nan, 0.0]],
        ]
    )
    states = hypotheses_module._encode_scale_states(
        adjacency_matrices, min_causal_effect=0.1
    )
    empty_states = hypotheses_module._encode_scale_states(
        np.empty((0, 2, 2)), min_causal_effect=0.1
    )

    assert states.shape == adjacency_matrices.shape
    assert states.dtype == np.int8
    assert empty_states.shape == (0, 2, 2)
    assert empty_states.dtype == np.int8


def test_graph_hash_deterministic_and_distinct():
    s1 = np.array([[0, 0], [1, 0]], dtype=np.int8)
    s2 = np.array([[0, 0], [1, 0]], dtype=np.int8)
    s3 = np.array([[0, 0], [0, 0]], dtype=np.int8)
    s4 = np.array([[0, 2], [2, 0]], dtype=np.int8)
    assert compute_graph_hash(s1) == compute_graph_hash(s2)
    assert compute_graph_hash(s1) != compute_graph_hash(s3)
    assert compute_graph_hash(s3) != compute_graph_hash(s4)


def test_edge_hypothesis_extracts_ordered_edges_and_canonical_arcs():
    result = make_bootstrap_result()
    extraction = EdgeHypothesis().extract(result)
    bp_long = extraction.bp_long

    edge01_s0 = bp_long[
        (bp_long.hypothesis_id == "edge:0->1") & (bp_long.scale_index == 0)
    ].iloc[0]
    assert edge01_s0.from_index == 0
    assert edge01_s0.to_index == 1
    assert edge01_s0.bp == pytest.approx(0.5)
    assert edge01_s0.bp_count == 2
    assert edge01_s0.nb == 4

    edge01_s1 = bp_long[
        (bp_long.hypothesis_id == "edge:0->1") & (bp_long.scale_index == 1)
    ].iloc[0]
    assert edge01_s1.bp == pytest.approx(1 / 3)
    assert set(extraction.bp_long.hypothesis_type) == {"edge", "arc"}
    assert extraction.bp_long.graph_hash.isna().all()

    arc_s0 = bp_long[
        (bp_long.hypothesis_id == "arc:0<->1") & (bp_long.scale_index == 0)
    ].iloc[0]
    assert arc_s0.from_index == 0
    assert arc_s0.to_index == 1
    assert arc_s0.bp == pytest.approx(0.25)

    # arc never observed at scale 1 -> explicit zero fill, not NaN
    arc_s1 = bp_long[
        (bp_long.hypothesis_id == "arc:0<->1") & (bp_long.scale_index == 1)
    ].iloc[0]
    assert arc_s1.bp == pytest.approx(0.0)
    arc_rows = bp_long[bp_long.hypothesis_type == "arc"]
    assert (arc_rows.from_index < arc_rows.to_index).all()


def test_edge_hypothesis_encodes_each_scale_once(monkeypatch):
    calls = []
    original_encode = hypotheses_module.encode_edge_states

    def record_encode(adjacency, min_causal_effect):
        calls.append(adjacency.shape)
        return original_encode(adjacency, min_causal_effect)

    monkeypatch.setattr(hypotheses_module, "encode_edge_states", record_encode)
    EdgeHypothesis().extract(make_bootstrap_result(), min_causal_effect=0.1)

    assert calls == [(4, 2, 2), (3, 2, 2)]


def test_encoding_failures_surface_during_hypothesis_extraction(monkeypatch):
    def fail_encode(adjacency, min_causal_effect):
        raise TypeError("synthetic encoding failure")

    monkeypatch.setattr(hypotheses_module, "encode_edge_states", fail_encode)

    with pytest.raises(TypeError, match="synthetic encoding failure"):
        EdgeHypothesis().extract(make_bootstrap_result())


def test_edge_hypothesis_zero_success_scale_uses_nan_for_both_types():
    result = make_bootstrap_result()
    result.scales.append(
        SingleScaleBootstrapResult(
            scale_index=2,
            scale_ratio=2.0,
            n_prime=200,
            sa=0.5,
            n_bootstrap_requested=1,
            n_success=0,
            adjacency_matrices=np.empty((0, 2, 2)),
            fit_failures=[],
            min_success_rate=0.5,
        )
    )

    zero_success_rows = EdgeHypothesis().extract(result).bp_long.query(
        "scale_index == 2"
    )
    assert set(zero_success_rows.hypothesis_type) == {"edge", "arc"}
    assert zero_success_rows.bp.isna().all()


def test_edge_hypothesis_reuses_result_at_different_min_causal_effects():
    result = make_bootstrap_result()
    low = EdgeHypothesis().extract(result, min_causal_effect=0.1)
    high = EdgeHypothesis().extract(result, min_causal_effect=0.3)

    low_bp = low.bp_long.query(
        "hypothesis_id == 'edge:0->1' and scale_index == 0"
    ).iloc[0].bp
    high_bp = high.bp_long.query(
        "hypothesis_id == 'edge:0->1' and scale_index == 0"
    ).iloc[0].bp
    assert low_bp == pytest.approx(0.5)
    assert high_bp == pytest.approx(0.0)


def test_graph_hypothesis_catalog_and_rows():
    extraction = GraphHypothesis().extract(make_bootstrap_result())
    graph_rows = extraction.bp_long[
        extraction.bp_long.hypothesis_type == "graph"
    ]
    assert graph_rows.from_index.isna().all()
    assert graph_rows.to_index.isna().all()
    assert graph_rows.graph_hash.notna().all()
    assert all(
        row.graph_hash in extraction.graph_states
        and compute_graph_hash(extraction.graph_states[row.graph_hash]) == row.graph_hash
        for row in graph_rows.itertuples()
    )
    assert set(extraction.bp_long.hypothesis_type) == {"graph"}


def test_graph_states_are_deduplicated_and_are_stable_copies():
    result = make_bootstrap_result()
    extraction = GraphHypothesis().extract(result)

    expected_hashes = {
        compute_graph_hash(
            encode_edge_states(adjacency, min_causal_effect=None)
        )
        for scale in result.scales
        for adjacency in scale.adjacency_matrices
    }
    assert set(extraction.graph_states) == expected_hashes
    assert len(extraction.graph_states) == 3
    for full_hash, state in extraction.graph_states.items():
        assert compute_graph_hash(state) == full_hash

    source_adjacency = result.scales[0].adjacency_matrices[0]
    source_state = encode_edge_states(source_adjacency, min_causal_effect=None)
    stored_state = extraction.graph_states[compute_graph_hash(source_state)]
    source_adjacency[1, 0] = 0.0
    assert stored_state[1, 0] == STATE_DIRECTED


def test_graph_catalog_preserves_bidirected_states():
    extraction = GraphHypothesis().extract(make_bootstrap_result())
    assert any(
        np.any(state == STATE_BIDIRECTED) for state in extraction.graph_states.values()
    )
    assert not extraction.bp_long.loc[
        extraction.bp_long.hypothesis_type == "graph", "graph_hash"
    ].isna().any()


def test_graph_hypothesis_fills_missing_scales():
    result = make_bootstrap_result()
    extraction = GraphHypothesis().extract(result)
    bp_long = extraction.bp_long

    arc_graph_hash = compute_graph_hash(np.array([[0, 2], [2, 0]], dtype=np.int8))
    arc_graph_id = f"graph:{arc_graph_hash}"

    # observed at scale 0 with bp=0.25
    row0 = bp_long[
        (bp_long.hypothesis_id == arc_graph_id) & (bp_long.scale_index == 0)
    ].iloc[0]
    assert row0.bp == pytest.approx(0.25)

    # not observed at scale 1 (n_success=3 > 0) -> explicit zero, not NaN
    row1 = bp_long[
        (bp_long.hypothesis_id == arc_graph_id) & (bp_long.scale_index == 1)
    ].iloc[0]
    assert row1.bp_count == 0
    assert row1.bp == pytest.approx(0.0)
    assert not np.isnan(row1.bp)

    assert row0.graph_hash == arc_graph_hash


def test_graph_hypothesis_reuses_result_with_threshold_specific_states():
    result = make_bootstrap_result()
    low = GraphHypothesis().extract(result, min_causal_effect=0.1)
    high = GraphHypothesis().extract(result, min_causal_effect=0.3)

    assert set(low.graph_states) != set(high.graph_states)
    assert all(
        not np.any(state == STATE_DIRECTED)
        for state in high.graph_states.values()
    )


@pytest.mark.parametrize("extractor", [EdgeHypothesis(), GraphHypothesis()])
def test_hypotheses_reject_negative_min_causal_effect(extractor):
    with pytest.raises(ValueError, match="min_causal_effect"):
        extractor.extract(make_bootstrap_result(), min_causal_effect=-0.01)


def test_graph_hypothesis_zero_success_scale_uses_nan_bp():
    result = make_bootstrap_result()
    result.scales.append(
        SingleScaleBootstrapResult(
            scale_index=2,
            scale_ratio=2.0,
            n_prime=200,
            sa=0.5,
            n_bootstrap_requested=1,
            n_success=0,
            adjacency_matrices=np.empty((0, 2, 2)),
            fit_failures=[],
            min_success_rate=0.5,
        )
    )
    bp_long = GraphHypothesis().extract(result).bp_long
    zero_success_rows = bp_long[bp_long.scale_index == 2]
    assert zero_success_rows.bp.isna().all()
