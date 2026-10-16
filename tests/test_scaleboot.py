import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from pyscaleboot import RScalebootClient, build_scaleboot_input


def make_bp_long():
    rows = []
    bps = {"signal": [0.99, 0.97, 0.92, 0.82, 0.65], "flat": [0.5] * 5}
    sa = [2.0, 1.4, 1.0, 0.7, 0.5]
    for hypothesis_id, bp in bps.items():
        for scale_index in range(5):
            rows.append(
                dict(
                    hypothesis_id=hypothesis_id,
                    scale_index=scale_index,
                    sa=sa[scale_index],
                    nb=5000,
                    bp=bp[scale_index],
                )
            )
    return pd.DataFrame(rows)


def test_build_scaleboot_input_pivots_hypotheses_and_scales():
    bp, scales = build_scaleboot_input(make_bp_long())

    assert list(bp.index) == ["flat", "signal"]
    assert list(bp.columns) == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(bp.loc["signal"], [0.99, 0.97, 0.92, 0.82, 0.65])
    assert list(scales.columns) == ["nb", "sa"]
    assert scales["sa"].tolist() == [2.0, 1.4, 1.0, 0.7, 0.5]
    assert scales["nb"].tolist() == [5000] * 5


def _rscript_with_scaleboot_available():
    if shutil.which("Rscript") is None:
        return False
    proc = subprocess.run(
        [
            "Rscript",
            "-e",
            "if (!requireNamespace('scaleboot', quietly=TRUE)) quit(status=1)",
        ],
        capture_output=True,
    )
    return proc.returncode == 0


requires_r_scaleboot = pytest.mark.skipif(
    not _rscript_with_scaleboot_available(),
    reason="Requires Rscript and the R 'scaleboot' package to be installed.",
)


@requires_r_scaleboot
def test_r_scaleboot_client_returns_au_si_pvalues():
    bp, scales = build_scaleboot_input(make_bp_long())

    result = RScalebootClient().fit(bp, scales, models=("poly.1",))

    assert list(result.columns) == [
        "hypothesis_id", "au_pvalue", "au_se", "si_pvalue", "si_se",
        "best_model", "beta0", "beta1",
    ]
    assert set(result["hypothesis_id"]) == {"signal", "flat"}
    assert result[["au_pvalue", "si_pvalue"]].stack().between(0, 1).all()
    p = result.set_index("hypothesis_id")["au_pvalue"]
    # A rising BP trend should have a higher AU p-value than the flat case.
    assert p["signal"] > p["flat"]


@requires_r_scaleboot
def test_r_scaleboot_client_accepts_edge_style_hypothesis_ids():
    bp_long = make_bp_long()
    bp_long["hypothesis_id"] = bp_long["hypothesis_id"].map(
        {"signal": "edge:0->1", "flat": "arc:0<->1"}
    )
    bp, scales = build_scaleboot_input(bp_long)

    result = RScalebootClient().fit(bp, scales, models=("poly.1",))

    assert set(result["hypothesis_id"]) == {"edge:0->1", "arc:0<->1"}


def test_r_scaleboot_client_reports_r_failure():
    bp, scales = build_scaleboot_input(make_bp_long())
    with pytest.raises(RuntimeError, match="scaleboot.r exited"):
        # A missing script should cause the client to report an R failure.
        RScalebootClient(script_path="nonexistent.r").fit(bp, scales)
