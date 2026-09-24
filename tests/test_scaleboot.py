import numpy as np
import pandas as pd
from lingam import DirectLiNGAM

from multiscale_bootstrap import EdgeHypothesis, GraphHypothesis, MultiScaleBootstrap
from multiscale_bootstrap._scaleboot import (
    ScalebootResult,
    ScalebootRow,
    ScalebootInput,
    build_scaleboot_input,
)
from multiscale_bootstrap._bootstrap import SingleScaleBootstrapResult

from ._helpers import make_bootstrap_result, make_chain_data


def test_python_pipeline_builds_scaleboot_input():
    result = MultiScaleBootstrap(
        DirectLiNGAM(),
        scale_ratios=(0.7, 1.0, 1.4),
        n_bootstrap=5,
        random_state=0,
    ).bootstrap(make_chain_data(n=300, seed=7))

    extraction = EdgeHypothesis().extract(result, min_causal_effect=0.15)
    scaleboot_input = build_scaleboot_input(extraction.bp_long)

    assert scaleboot_input.hypothesis_ids
    assert scaleboot_input.bp_matrix.shape == (
        len(scaleboot_input.hypothesis_ids),
        len(scaleboot_input.sa),
    )


def test_build_scaleboot_input_and_scaleboot_result_preserve_ids():
    result = make_bootstrap_result()
    extraction = EdgeHypothesis().extract(result)
    scaleboot_input = build_scaleboot_input(
        extraction.bp_long
    )

    assert "edge:0->1" in scaleboot_input.hypothesis_ids
    assert "arc:0<->1" in scaleboot_input.hypothesis_ids
    assert scaleboot_input.bp_matrix.shape[0] == len(scaleboot_input.hypothesis_ids)
    assert scaleboot_input.bp_matrix.shape[1] == len(scaleboot_input.sa)

    rows = [
        ScalebootRow(
            hypothesis_id=hid,
            au_pvalue=0.5,
            au_se=None,
            si_pvalue=0.5,
            si_se=None,
            selected_model="poly.1",
            beta0=None,
            beta1=None,
        )
        for hid in scaleboot_input.hypothesis_ids
    ]
    test_result = ScalebootResult(rows=rows)
    result_frame = test_result.to_frame()
    assert list(result_frame.hypothesis_id) == scaleboot_input.hypothesis_ids
    assert set(result_frame.columns) == {
        "hypothesis_id",
        "au_pvalue",
        "au_se",
        "si_pvalue",
        "si_se",
        "selected_model",
        "beta0",
        "beta1",
    }


def test_build_scaleboot_input_handles_mixed_pairwise_hypotheses():
    extraction = EdgeHypothesis().extract(make_bootstrap_result())
    scaleboot_input = build_scaleboot_input(extraction.bp_long)

    assert set(extraction.bp_long.hypothesis_type) == {"edge", "arc"}
    assert len(scaleboot_input.hypothesis_ids) == len(
        set(scaleboot_input.hypothesis_ids)
    )
    assert {"edge:0->1", "arc:0<->1"}.issubset(
        scaleboot_input.hypothesis_ids
    )


def test_build_scaleboot_input_includes_every_graph_hypothesis():
    result = make_bootstrap_result()
    extraction = GraphHypothesis().extract(result)
    scaleboot_input = build_scaleboot_input(
        extraction.bp_long
    )
    graph_ids = {h for h in scaleboot_input.hypothesis_ids if h.startswith("graph:")}
    expected_graph_ids = set(
        extraction.bp_long.loc[
            extraction.bp_long["hypothesis_type"] == "graph", "hypothesis_id"
        ]
    )
    assert graph_ids == expected_graph_ids


def test_build_scaleboot_input_models_default_and_override():
    result = make_bootstrap_result()
    extraction = EdgeHypothesis().extract(result)

    default_input = build_scaleboot_input(
        extraction.bp_long
    )
    assert default_input.models == (
        "poly.1",
        "poly.2",
        "poly.3",
        "sing.3",
        "sphe.3",
    )
    assert default_input.k == 3
    assert default_input.sk == 3
    assert default_input.select == "average"

    custom_input = build_scaleboot_input(
        extraction.bp_long,
        models=("poly.1", "poly.2", "poly.3"),
        k=2,
        sk=2,
        select="best",
    )
    assert custom_input.models == ("poly.1", "poly.2", "poly.3")
    assert custom_input.k == 2
    assert custom_input.sk == 2
    assert custom_input.select == "best"


def test_build_scaleboot_input_rejects_duplicate_hypothesis_scale_rows():
    result = make_bootstrap_result()
    extraction = EdgeHypothesis().extract(result)
    duplicate_rows = pd.concat(
        [extraction.bp_long, extraction.bp_long.iloc[[0]]], ignore_index=True
    )
    with np.testing.assert_raises(ValueError):
        build_scaleboot_input(duplicate_rows)


def test_zero_success_scale_is_invalid_at_zero_min_success_rate():
    scale = SingleScaleBootstrapResult(
        scale_index=1,
        scale_ratio=1.0,
        n_prime=100,
        sa=1.0,
        n_bootstrap_requested=10,
        n_success=0,
        adjacency_matrices=np.empty((0, 2, 2)),
        fit_failures=[],
        min_success_rate=0.0,
    )

    assert not scale.is_valid


def test_build_scaleboot_input_rejects_zero_nb_for_valid_scale():
    result = make_bootstrap_result()
    extraction = EdgeHypothesis().extract(result)
    bp_long = extraction.bp_long.copy()
    bp_long.loc[bp_long["is_valid_scale"], "nb"] = 0

    with np.testing.assert_raises_regex(ValueError, "positive integers"):
        build_scaleboot_input(bp_long)


def test_empty_scaleboot_result_has_stable_schema():
    result = ScalebootResult(rows=[]).to_frame()

    assert list(result.columns) == [
        "hypothesis_id",
        "au_pvalue",
        "au_se",
        "si_pvalue",
        "si_se",
        "selected_model",
        "beta0",
        "beta1",
    ]


