"""Real-R integration tests for RScalebootClient.

Skipped automatically when Rscript or the scaleboot R package are not
available (e.g. in CI), so this file is safe to keep in the normal test
suite alongside the R-free tests.
"""

import shutil
import subprocess

import numpy as np
import pytest
from lingam import DirectLiNGAM

from multiscale_bootstrap import (
    MultiScaleBootstrap,
    EdgeHypothesis,
)
from multiscale_bootstrap._scaleboot import RScalebootClient, ScalebootInput, build_scaleboot_input

from ._helpers import make_chain_data


def _rscript_with_scaleboot_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    try:
        proc = subprocess.run(
            ["Rscript", "-e", "if (!requireNamespace('scaleboot', quietly=TRUE)) quit(status=1)"],
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    return proc.returncode == 0


requires_r_scaleboot = pytest.mark.skipif(
    not _rscript_with_scaleboot_available(),
    reason="Requires Rscript and the R 'scaleboot' package to be installed.",
)


@requires_r_scaleboot
def test_r_scaleboot_client_returns_valid_probabilities():
    sa = np.array([2.0, 1.4, 1.0, 0.7, 0.5])
    nb = np.full(5, 5000)
    # a clear high-bp signal, a flat no-signal case, and both degenerate extremes
    bp_signal = np.array([0.99, 0.97, 0.92, 0.82, 0.65])
    bp_flat = np.full(5, 0.5)
    bp_zero = np.zeros(5)
    bp_one = np.ones(5)

    # poly.1 only (explicitly selected): with just 5 scales,
    # adding poly.2 / poly.3 candidates is numerically unstable (see
    # test_r_scaleboot_client_multi_model_can_be_unstable_with_few_scales).
    scaleboot_input = ScalebootInput(
        hypothesis_ids=["signal", "flat", "zero", "one"],
        bp_matrix=np.vstack([bp_signal, bp_flat, bp_zero, bp_one]),
        nb=nb,
        sa=sa,
        models=("poly.1",),
        k=3,
        sk=3,
        select="average",
    )

    result = RScalebootClient().fit(scaleboot_input)

    # row count and hypothesis_id order must match the input exactly, since
    # result.csv carries no hypothesis_id column of its own.
    result_frame = result.to_frame()
    assert list(result_frame.hypothesis_id) == scaleboot_input.hypothesis_ids

    rows_by_id = {r.hypothesis_id: r for r in result.rows}

    for hid in scaleboot_input.hypothesis_ids:
        row = rows_by_id[hid]
        assert 0.0 <= row.au_pvalue <= 1.0
        assert 0.0 <= row.si_pvalue <= 1.0

    # bp increasing with scale -> larger AU p-value than the flat/no-signal case
    assert rows_by_id["signal"].au_pvalue > rows_by_id["flat"].au_pvalue
    assert rows_by_id["zero"].au_pvalue == pytest.approx(0.0, abs=1e-6)
    assert rows_by_id["one"].au_pvalue == pytest.approx(1.0, abs=1e-6)


@requires_r_scaleboot
def test_r_scaleboot_client_multi_model_can_be_unstable_with_few_scales():
    """Regression/documentation test for a real finding from Step 6 validation.

    With only 5-10 scales, letting scaleboot choose between poly.1/poly.2/
    poly.3 by AIC can produce a wildly different (and untrustworthy)
    au_pvalue than the poly.1-only fit, because the higher-order models
    nearly interpolate the handful of scale points and extrapolate badly
    at sigma=1. This is why a conservative ``models=("poly.1",)`` can be
    passed explicitly rather than using scaleboot's multi-model default.
    """
    sa = np.array([2.0, 1.4, 1.0, 0.7, 0.5])
    nb = np.full(5, 5000)
    bp_signal = np.array([0.99, 0.97, 0.92, 0.82, 0.65])

    def au_pvalue_for(models):
        scaleboot_input = ScalebootInput(
            hypothesis_ids=["signal"],
            bp_matrix=bp_signal.reshape(1, -1),
            nb=nb,
            sa=sa,
            models=models,
            k=3,
            sk=3,
            select="average",
        )
        row = RScalebootClient().fit(scaleboot_input).rows[0]
        return row.au_pvalue

    au_poly1 = au_pvalue_for(("poly.1",))
    au_poly1_2_3 = au_pvalue_for(("poly.1", "poly.2", "poly.3"))

    # poly.1-only stays close to the raw bp trend (clearly "significant").
    assert au_poly1 > 0.5
    # adding higher-order models flips this to a near-zero p-value with
    # this few scales - a large, order-of-magnitude discrepancy.
    assert au_poly1_2_3 < 0.01


@requires_r_scaleboot
def test_r_scaleboot_client_missing_rscript_raises_runtime_error():
    sa = np.array([1.0, 2.0])
    scaleboot_input = ScalebootInput(
        hypothesis_ids=["h"],
        bp_matrix=np.array([[0.5, 0.5]]),
        nb=np.array([10, 10]),
        sa=sa,
        models=("poly.1",),
        k=3,
        sk=3,
        select="average",
    )
    client = RScalebootClient(rscript_path="Rscript-does-not-exist")
    with pytest.raises(RuntimeError, match="Could not run"):
        client.fit(scaleboot_input)


@requires_r_scaleboot
def test_r_scaleboot_client_honors_custom_models_k_sk_select():
    """``models``/``k``/``sk``/``select`` set via build_scaleboot_input must
    reach Rscript, not just the default models/``select="average"`` values.
    """
    sa = 9.0 ** np.linspace(-1, 1, 9)  # enough, closely-spaced scales for poly.3
    nb = np.full(9, 5000)
    bp_signal = 1.0 / (1.0 + np.exp(-(np.log(sa) * -3.0)))

    scaleboot_input = ScalebootInput(
        hypothesis_ids=["signal"],
        bp_matrix=bp_signal.reshape(1, -1),
        nb=nb,
        sa=sa,
        models=("poly.1", "poly.2", "poly.3"),
        k=2,
        sk=2,
        select="best",
    )

    r_result = RScalebootClient().fit(scaleboot_input)
    row = r_result.rows[0]
    assert 0.0 <= row.au_pvalue <= 1.0
    # select="best" reports one concrete model name, not "average".
    assert row.selected_model in scaleboot_input.models


@requires_r_scaleboot
def test_r_scaleboot_client_end_to_end_pipeline_with_direct_lingam():
    X = make_chain_data(n=400, seed=7)
    msb = MultiScaleBootstrap(
        DirectLiNGAM(),
        scale_ratios=(0.5, 0.7, 1.0, 1.4, 2.0),
        n_bootstrap=15,
        random_state=0,
    )
    result = msb.bootstrap(X, variable_names=["x0", "x1", "x2"])

    extraction = EdgeHypothesis().extract(result, min_causal_effect=0.15)
    scaleboot_input = build_scaleboot_input(
        extraction.bp_long
    )

    r_result = RScalebootClient().fit(scaleboot_input)
    assert len(r_result.rows) == len(scaleboot_input.hypothesis_ids)
    assert [r.hypothesis_id for r in r_result.rows] == scaleboot_input.hypothesis_ids

    result_frame = r_result.to_frame()
    assert list(result_frame.hypothesis_id) == scaleboot_input.hypothesis_ids
    assert (result_frame["au_pvalue"].between(0, 1)).all()
    assert (result_frame["si_pvalue"].between(0, 1)).all()

