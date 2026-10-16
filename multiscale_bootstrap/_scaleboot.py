"""Prepare bootstrap results for R scaleboot AU/SI p-value calculation."""

import os
import subprocess
import tempfile

import pandas as pd


def build_scaleboot_input(bp_long):
    """Pivot ``bp_long`` into the inputs of ``scaleboot::sbfit``.

    Parameters
    ----------
    bp_long : pandas.DataFrame
        Long-format bootstrap probabilities with the columns
        ``hypothesis_id``, ``scale_index``, ``sa``, ``nb`` and ``bp``, as
        returned by the hypothesis extractors. Every hypothesis in
        ``bp_long`` is included; filter the table beforehand to select
        hypotheses.

    Returns
    -------
    bp : pandas.DataFrame, shape (n_hypotheses, n_scales)
        Bootstrap probabilities. The index is ``hypothesis_id`` and the
        columns are ``scale_index``.
    scales : pandas.DataFrame, shape (n_scales, 2)
        Number of successful replicates ``nb`` and scale ``sa`` of each
        column of ``bp``.
    """
    bp = bp_long.pivot(index="hypothesis_id", columns="scale_index", values="bp")
    scales = (
        bp_long[["scale_index", "nb", "sa"]]
        .drop_duplicates("scale_index")
        .set_index("scale_index")
        .loc[bp.columns]
    )
    return bp, scales


class RScalebootClient:
    """Real ``scaleboot::sbfit`` via ``Rscript`` + CSV file exchange.

    Follows the same ``--key=value`` + ``temp_dir`` CSV convention used by
    ``examples/pruning_missing_data``. Requires ``Rscript`` on ``PATH`` and
    the R ``scaleboot`` package installed.

    Parameters
    ----------
    rscript_path : str, optional
        Executable used to invoke R. Defaults to ``"Rscript"``.
    script_path : str, optional
        Path to the bundled ``scaleboot.r`` script. If omitted, the
        script next to this module is used.
    """

    def __init__(self, rscript_path="Rscript", script_path=None):
        self.rscript_path = rscript_path
        self.script_path = script_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "scaleboot.r"
        )

    def fit(
        self,
        bp,
        scales,
        models=("poly.1", "poly.2", "poly.3", "sing.3", "sphe.3"),
        k=3,
        sk=3,
        select="average",
    ):
        """Run scaleboot and return the AU/SI p-values.

        Parameters
        ----------
        bp, scales : pandas.DataFrame
            Outputs of :func:`build_scaleboot_input`.
        models : tuple of str, optional
            Candidate models handed to ``scaleboot::sbfit``. The default is
            scaleboot's own default. With few scales, AIC-based selection
            among several models can be unstable; a conservative choice such
            as ``("poly.1",)`` is then recommended.
        k, sk : int, optional
            Orders used to report the AU (``k``) and SI (``sk``) p-values.
        select : {"average", "best"}, optional
            How multiple models are combined.

        Returns
        -------
        pandas.DataFrame
            One row per hypothesis with the columns ``hypothesis_id``,
            ``au_pvalue``, ``au_se``, ``si_pvalue``, ``si_se``,
            ``best_model``, ``beta0`` and ``beta1``.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            bp.to_csv(os.path.join(temp_dir, "bp.csv"), header=False)
            scales[["nb", "sa"]].to_csv(
                os.path.join(temp_dir, "scales.csv"), index=False
            )

            args = [
                f"--temp_dir={temp_dir}",
                f"--models={','.join(models)}",
                f"--k={k}",
                f"--sk={sk}",
                f"--select={select}",
            ]
            proc = subprocess.run(
                [self.rscript_path, self.script_path, *args],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"scaleboot.r exited with status {proc.returncode}.\n"
                    f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
                )

            return pd.read_csv(os.path.join(temp_dir, "result.csv"))
