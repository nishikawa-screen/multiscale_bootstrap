"""Prepare bootstrap results for R scaleboot AU/SI p-value calculation."""

import os
import subprocess
import shutil
import tempfile

import numpy as np
import pandas as pd


class ScalebootInput:
    """Input matrix and scale metadata passed to scaleboot.

    Parameters
    ----------
    hypothesis_ids : list of str
        Hypothesis identifiers in the same order as the rows of
        ``bp_matrix``.
    bp_matrix : ndarray, shape (n_hypotheses, n_scales)
        Bootstrap probabilities for each hypothesis and valid scale.
    nb : ndarray, shape (n_scales,)
        Number of successful bootstrap replicates for each scale.
    sa : ndarray, shape (n_scales,)
        Scale values passed to scaleboot.
    models : tuple of str
        Candidate scaleboot models.
    k, sk : int
        Scale indices used for AU and SI p-values.
    select : str
        Model-combination strategy used by scaleboot.

    Attributes
    ----------
    hypothesis_ids : list of str
        Hypothesis identifiers.
    bp_matrix : ndarray
        Bootstrap-probability matrix.
    nb, sa : ndarray
        Replicate counts and scale values.
    models : tuple of str
        Candidate scaleboot models.
    k, sk : int
        AU and SI scale indices.
    select : str
        Model-combination strategy.
    """

    def __init__(self, hypothesis_ids, bp_matrix, nb, sa, models, k, sk, select):
        """Construct scaleboot input data."""
        self.hypothesis_ids = hypothesis_ids
        self.bp_matrix = bp_matrix
        self.nb = nb
        self.sa = sa
        self.models = models
        self.k = k
        self.sk = sk
        self.select = select


class ScalebootRow:
    """AU/SI results returned by scaleboot for one hypothesis.

    Parameters
    ----------
    hypothesis_id : str
        Hypothesis identifier.
    au_pvalue, au_se : float
        AU p-value and its standard error.
    si_pvalue, si_se : float
        SI p-value and its standard error.
    selected_model : str
        Model selected or reported by scaleboot.
    beta0, beta1 : float
        Fitted boundary parameters reported by scaleboot.
    """

    def __init__(
        self,
        hypothesis_id,
        au_pvalue,
        au_se,
        si_pvalue,
        si_se,
        selected_model,
        beta0,
        beta1,
    ):
        """Construct one scaleboot result row."""
        self.hypothesis_id = hypothesis_id
        self.au_pvalue = au_pvalue
        self.au_se = au_se
        self.si_pvalue = si_pvalue
        self.si_se = si_se
        self.selected_model = selected_model
        self.beta0 = beta0
        self.beta1 = beta1


class ScalebootResult:
    """AU/SI results for all submitted hypotheses.

    Parameters
    ----------
    rows : list of ScalebootRow
        Results in the same hypothesis order submitted to scaleboot.
    """

    def __init__(self, rows):
        """Construct a scaleboot result.

        Parameters
        ----------
        rows : list of ScalebootRow
            Results in hypothesis submission order.
        """
        self.rows = rows

    def to_frame(self):
        """Convert the result rows to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per hypothesis with stable output columns.
        """
        columns = [
            "hypothesis_id",
            "au_pvalue",
            "au_se",
            "si_pvalue",
            "si_se",
            "selected_model",
            "beta0",
            "beta1",
        ]
        return pd.DataFrame([r.__dict__ for r in self.rows], columns=columns)


_REQUIRED_BP_COLUMNS = {
    "hypothesis_id",
    "scale_index",
    "sa",
    "nb",
    "bp",
    "is_valid_scale",
}
_SCALE_METADATA_COLUMNS = ["scale_index", "sa", "nb", "is_valid_scale"]


def _validate_required_columns(bp_long, required_columns):
    """Raise an error when required columns are absent.

    Parameters
    ----------
    bp_long : pandas.DataFrame
        Bootstrap-probability table to validate.
    required_columns : set of str
        Required column names.
    """
    missing = sorted(required_columns.difference(bp_long.columns))
    if missing:
        raise ValueError(
            "bp_long is missing required columns: " + ", ".join(missing)
        )


def _validate_duplicate_rows(bp_long):
    """Reject duplicate hypothesis and scale combinations.

    Parameters
    ----------
    bp_long : pandas.DataFrame
        Bootstrap-probability table to validate.

    Raises
    ------
    ValueError
        If a hypothesis has more than one row for the same scale.
    """
    duplicate_mask = bp_long.duplicated(
        ["hypothesis_id", "scale_index"], keep=False
    )
    if duplicate_mask.any():
        raise ValueError(
            "bp_long contains duplicate (hypothesis_id, scale_index) rows; "
            "the scaleboot pivot would be ambiguous."
        )


def _validated_scale_metadata(bp_long):
    """Extract and validate metadata for valid scales.

    Parameters
    ----------
    bp_long : pandas.DataFrame
        Bootstrap-probability table containing scale metadata.

    Returns
    -------
    pandas.DataFrame
        Valid scale metadata sorted by ``scale_index``. The ``sa`` column is
        float-valued and the ``nb`` column contains positive integers.
    """
    scale_metadata = (
        bp_long[_SCALE_METADATA_COLUMNS]
        .drop_duplicates()
        .sort_values("scale_index")
        .reset_index(drop=True)
    )
    inconsistent = scale_metadata[
        scale_metadata.duplicated("scale_index", keep=False)
    ]["scale_index"].drop_duplicates().tolist()
    if inconsistent:
        raise ValueError(
            "Inconsistent scale metadata for scale_index values: "
            + ", ".join(str(value) for value in sorted(inconsistent))
        )
    valid_metadata = scale_metadata[
        scale_metadata["is_valid_scale"].fillna(False).astype(bool)
    ].copy()
    if valid_metadata.empty:
        raise ValueError("bp_long contains no valid scales.")

    try:
        sa_vec = valid_metadata["sa"].to_numpy(dtype=float)
        nb_as_float = valid_metadata["nb"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Valid scale metadata must contain numeric sa and nb."
        ) from exc
    if not np.isfinite(sa_vec).all():
        raise ValueError("Valid scale metadata contains non-finite sa values.")
    if (
        not np.isfinite(nb_as_float).all()
        or (nb_as_float <= 0).any()
        or (nb_as_float != np.floor(nb_as_float)).any()
    ):
        raise ValueError(
            "Valid scale metadata nb values must be finite positive integers."
        )
    valid_metadata["sa"] = sa_vec
    valid_metadata["nb"] = nb_as_float.astype(int)
    return valid_metadata


def build_scaleboot_input(
    bp_long,
    models=("poly.1", "poly.2", "poly.3", "sing.3", "sphe.3"),
    k=3,
    sk=3,
    select="average",
    hypothesis_ids=None,
):
    """Pivot ``bp_long`` into a scaleboot-ready matrix for every hypothesis.

    Every hypothesis present in ``bp_long`` is included; there is no top-k /
    min-bp filtering (AU/SI fitting is cheap per hypothesis, see the README
    for the rationale).

    Parameters
    ----------
    models : tuple of str, optional
        Candidate models handed to ``scaleboot::sbfit``. Valid names are
        ``"poly.m"`` (m >= 1), ``"sing.m"`` (m >= 3) and ``"sphe.3"``, and
        must be given in increasing order (``poly.(m-1)`` before
        ``poly.m``, ``poly.(m-1)``/``sing.(m-1)`` before ``sing.m``).
        The default matches scaleboot's own default,
        ``sboptions()$models`` =
        ``("poly.1", "poly.2", "poly.3", "sing.3", "sphe.3")``. When more
        than one model is given, scaleboot fits all of them and selects/
        averages by AIC (``select``), reporting the winner in
        ``selected_model``. If AIC-based selection is unstable for your
        setup - e.g. with the 5-10 widely-spaced scales this pipeline often
        uses - pass a smaller, more conservative set such as
        ``models=("poly.1",)`` or
        ``models=("poly.1", "poly.2", "poly.3")``. See the README for
        the trade-offs.
    k, sk : int, optional
        Scale indices (1-based) used to report the AU (``k``) and SI
        (``sk``) p-values in ``sbpval()``.
    select : {"average", "best"}, optional
        How ``sbpval()`` combines multiple ``models``.
    hypothesis_ids : sequence of str, optional
        Subset of hypothesis identifiers to include. By default, all
        hypotheses in ``bp_long`` are included.

    Returns
    -------
    ScalebootInput
        Matrix, scale metadata, and fitting options for scaleboot.

    Raises
    ------
    ValueError
        If the table is empty, malformed, contains invalid scale metadata,
        or lacks a bootstrap probability for a selected hypothesis and valid
        scale.
    """

    _validate_required_columns(bp_long, _REQUIRED_BP_COLUMNS)
    if bp_long.empty:
        raise ValueError("bp_long must not be empty.")
    _validate_duplicate_rows(bp_long)
    valid_metadata = _validated_scale_metadata(bp_long)
    sa_vec = valid_metadata["sa"].to_numpy()
    nb_vec = valid_metadata["nb"].to_numpy()
    valid_scale_indices = valid_metadata["scale_index"].tolist()

    if hypothesis_ids is None:
        selected_ids = (
            bp_long[["hypothesis_id"]]
            .drop_duplicates()
            .sort_values(["hypothesis_id"])["hypothesis_id"]
            .tolist()
        )
    else:
        selected_ids = list(hypothesis_ids)
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("hypothesis_ids contains duplicate IDs.")
        known_ids = set(bp_long["hypothesis_id"])
        unknown_ids = sorted(set(selected_ids).difference(known_ids))
        if unknown_ids:
            raise ValueError(
                "Unknown hypothesis_ids: " + ", ".join(str(value) for value in unknown_ids)
            )
    if not selected_ids:
        raise ValueError("At least one hypothesis must be selected.")

    valid_bp_long = bp_long[bp_long["scale_index"].isin(valid_scale_indices)]
    pivot = valid_bp_long[valid_bp_long["hypothesis_id"].isin(selected_ids)].pivot(
        index="hypothesis_id", columns="scale_index", values="bp"
    )
    pivot = pivot.reindex(index=selected_ids, columns=valid_scale_indices)

    missing = pivot.isna()
    if missing.any().any():
        affected = [
            f"{hypothesis_id}@{scale_index}"
            for hypothesis_id, row in missing.iterrows()
            for scale_index, is_missing in row.items()
            if is_missing
        ]
        raise ValueError(
            "Missing BP values for valid hypothesis/scale combinations: "
            + ", ".join(affected)
        )

    bp_matrix = pivot.to_numpy(dtype=float)
    if not np.isfinite(bp_matrix).all():
        raise ValueError("BP values must be finite on valid scales.")
    if ((bp_matrix < 0.0) | (bp_matrix > 1.0)).any():
        raise ValueError("BP values must be within [0.0, 1.0].")

    return ScalebootInput(
        hypothesis_ids=selected_ids,
        bp_matrix=bp_matrix,
        nb=nb_vec,
        sa=sa_vec,
        models=tuple(models),
        k=k,
        sk=sk,
        select=select,
    )


class RScalebootClient:
    """Real ``scaleboot::sbfit`` via ``Rscript`` + CSV file exchange.

    Follows the same ``--key=value`` + ``temp_dir`` CSV convention used by
    ``examples/pruning_missing_data`` (``impute_by_mice`` /
    ``estimate_adj_matrix_for_stacked_imputed_datasets``), not ``rpy2``.
    Requires ``Rscript`` on ``PATH`` and the R ``scaleboot`` package
    installed.

    Parameters
    ----------
    rscript_path : str, optional
        Executable used to invoke R. Defaults to ``"Rscript"``.
    script_path : str, optional
        Path to the bundled ``scaleboot_fit.r`` script. If omitted, the
        script next to this module is used.
    """

    def __init__(self, rscript_path="Rscript", script_path=None):
        """Construct an R scaleboot client.

        Parameters
        ----------
        rscript_path : str, optional
            Rscript executable name or path.
        script_path : str, optional
            Path to the scaleboot wrapper script.
        """
        self.rscript_path = rscript_path
        self.script_path = script_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "scaleboot_fit.r"
        )

    def fit(self, scaleboot_input):
        """Run scaleboot for a prepared input.

        Parameters
        ----------
        scaleboot_input : ScalebootInput
            Input matrix, scale metadata, and scaleboot options.

        Returns
        -------
        ScalebootResult
            AU/SI results in the input hypothesis order.

        Raises
        ------
        RuntimeError
            If Rscript cannot be started, the R script fails, or its output
            does not contain one row per submitted hypothesis.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            bp_df = pd.DataFrame(
                scaleboot_input.bp_matrix, index=scaleboot_input.hypothesis_ids
            )
            bp_df.to_csv(os.path.join(temp_dir, "bp.csv"), header=False)

            pd.DataFrame({"nb": scaleboot_input.nb, "sa": scaleboot_input.sa}).to_csv(
                os.path.join(temp_dir, "scales.csv"), index=False
            )

            args = [
                f"--temp_dir={temp_dir}",
                f"--models={','.join(scaleboot_input.models)}",
                f"--k={scaleboot_input.k}",
                f"--sk={scaleboot_input.sk}",
                f"--select={scaleboot_input.select}",
            ]
            try:
                proc = subprocess.run(
                    [self.rscript_path, self.script_path, *args],
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Could not run Rscript executable '{self.rscript_path}'. "
                    "Verify rscript_path and install R if necessary."
                ) from exc

            if proc.returncode != 0:
                raise RuntimeError(
                    f"scaleboot_fit.r exited with status {proc.returncode}.\n"
                    f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
                )

            out_df = pd.read_csv(os.path.join(temp_dir, "result.csv"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if len(out_df) != len(scaleboot_input.hypothesis_ids):
            raise RuntimeError(
                "scaleboot_fit.r result row count "
                f"({len(out_df)}) does not match the number of hypotheses "
                f"submitted ({len(scaleboot_input.hypothesis_ids)})."
            )

        # result.csv has no hypothesis_id column; rows are in input order.
        rows = [
            ScalebootRow(
                hypothesis_id=hid,
                au_pvalue=row.get("au_pvalue"),
                au_se=row.get("au_se"),
                si_pvalue=row.get("si_pvalue"),
                si_se=row.get("si_se"),
                selected_model=row.get("selected_model"),
                beta0=row.get("beta0"),
                beta1=row.get("beta1"),
            )
            for hid, (_, row) in zip(scaleboot_input.hypothesis_ids, out_df.iterrows())
        ]
        return ScalebootResult(rows=rows)


