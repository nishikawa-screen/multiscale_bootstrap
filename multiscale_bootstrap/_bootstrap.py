"""Core multiscale bootstrap engine for LiNGAM-compatible models."""

import copy

import numpy as np
import pandas as pd

# Absolute floor: even the 2-parameter poly.1 scaleboot model needs at
# least this many valid scales to fit; not user-configurable.
_MIN_VALID_SCALES = 2


def _min_n_prime(d):
    """Return the minimum acceptable resampled size.

    Parameters
    ----------
    d : int
        Number of variables.

    Returns
    -------
    int
        Minimum number of rows required for a resampled dataset.
    """
    return max(2 * d + 1, 30)


def _resolve_scale(n, scale_ratio, d):
    """Resolve the actual resampled size and ``sa`` for one scale.

    Parameters
    ----------
    n : int
        Original sample size.
    scale_ratio : float
        Requested ratio ``n_prime / n``.
    d : int
        Number of variables.

    Returns
    -------
    n_prime : int
        Rounded resampled sample size.
    sa : float
        ``n / n_prime`` computed from the *rounded* ``n_prime``.
    """
    n_prime = int(round(scale_ratio * n))
    min_np = _min_n_prime(d)
    if n_prime < min_np:
        raise ValueError(
            f"scale_ratio={scale_ratio} produces n_prime={n_prime}, but "
            f"at least {min_np} resampled rows are required for d={d} "
            "variables. Increase scale_ratio or n, or reduce the number of "
            "variables."
        )
    sa = n / n_prime
    return n_prime, sa


class FitFailure:
    """Record of a single failed bootstrap replicate.

    Parameters
    ----------
    replicate_index : int
        Index of the failed replicate.
    error_type : str
        Exception class name.
    error_message : str
        Exception message.
    """

    def __init__(self, replicate_index, error_type, error_message):
        """Construct a failed-replicate record.

        Parameters
        ----------
        replicate_index : int
            Index of the failed replicate.
        error_type : str
            Exception class name.
        error_message : str
            Exception message.
        """
        self.replicate_index = replicate_index
        self.error_type = error_type
        self.error_message = error_message


class FitOutcome:
    """Outcome of a single model-fitting attempt.

    Parameters
    ----------
    adjacency : ndarray, optional
        Extracted adjacency matrix when fitting succeeds.
    error_type : str, optional
        Exception class name when fitting fails.
    error_message : str, optional
        Exception message when fitting fails.
    """

    def __init__(self, adjacency=None, error_type=None, error_message=None):
        """Construct a fitting outcome.

        Parameters
        ----------
        adjacency : ndarray, optional
            Extracted adjacency matrix on success.
        error_type : str, optional
            Exception class name on failure.
        error_message : str, optional
            Exception message on failure.
        """
        self.adjacency = adjacency
        self.error_type = error_type
        self.error_message = error_message

    @property
    def ok(self):
        """Whether the model fitting and matrix extraction succeeded."""
        return self.adjacency is not None


class SingleScaleBootstrapResult:
    """Bootstrap outcome for one scale (one value of ``n_prime`` / ``sa``).

    ``n`` (original sample size) is not stored here; it is the same for
    every scale, so use the parent ``MultiScaleBootstrapResult.n`` instead.
    ``adjacency_matrices`` retains the raw fitted coefficients and ``NaN``
    markers. Edge presence is determined later during hypothesis extraction.

    Parameters
    ----------
    scale_index : int
        Index of this scale.
    scale_ratio : float
        Requested ratio ``n_prime / n``.
    n_prime : int
        Actual resampled sample size.
    sa : float
        ``n / n_prime`` for the actual sample size.
    n_bootstrap_requested : int
        Number of requested replicates.
    n_success : int
        Number of successful replicates.
    adjacency_matrices : ndarray, shape (n_success, d, d)
        Raw fitted coefficients and NaN markers.
    fit_failures : list of FitFailure
        Failed replicate details.
    min_success_rate : float, optional
        Minimum success rate required for a valid scale.
    """

    def __init__(
        self,
        scale_index,
        scale_ratio,
        n_prime,
        sa,
        n_bootstrap_requested,
        n_success,
        adjacency_matrices,
        fit_failures,
        min_success_rate=0.5,
    ):
        """Construct the result for one bootstrap scale.

        Parameters
        ----------
        scale_index : int
            Index of this scale.
        scale_ratio : float
            Requested ratio ``n_prime / n``.
        n_prime : int
            Actual resampled sample size.
        sa : float
            ``n / n_prime`` for the actual sample size.
        n_bootstrap_requested : int
            Number of requested replicates.
        n_success : int
            Number of successful replicates.
        adjacency_matrices : ndarray
            Raw fitted adjacency matrices.
        fit_failures : list of FitFailure
            Failed replicate details.
        min_success_rate : float, optional
            Minimum success rate required for validity.
        """
        self.scale_index = scale_index
        self.scale_ratio = scale_ratio
        self.n_prime = n_prime
        self.sa = sa
        self.n_bootstrap_requested = n_bootstrap_requested
        self.n_success = n_success
        self.adjacency_matrices = adjacency_matrices
        self.fit_failures = fit_failures
        self.min_success_rate = min_success_rate

    @property
    def is_valid(self):
        """Whether this scale meets the configured success-rate threshold."""
        if self.n_bootstrap_requested <= 0 or self.n_success <= 0:
            return False
        success_rate = self.n_success / self.n_bootstrap_requested
        return success_rate >= self.min_success_rate


class MultiScaleBootstrapResult:
    """Aggregate result across all scales (bootstrap replicates only).

    The result holds raw adjacency matrices before thresholding. Hypothesis
    extractors apply ``min_causal_effect`` when deciding edge presence, so the
    same bootstrap result can be re-aggregated with different thresholds
    without rerunning the bootstrap.

    Parameters
    ----------
    scales : list of SingleScaleBootstrapResult
        Results for each scale.
    n : int
        Original sample size.
    variable_names : list of str
        Variable names.
    """

    def __init__(self, scales, n, variable_names):
        """Construct an aggregate multi-scale bootstrap result.

        Parameters
        ----------
        scales : list of SingleScaleBootstrapResult
            Result for each requested scale.
        n : int
            Original sample size.
        variable_names : list of str
            Variable names.
        """
        self.scales = scales
        self.n = n
        self.variable_names = variable_names

    def scale_summary(self):
        """Return a summary table containing one row for each scale.

        Returns
        -------
        pandas.DataFrame
            Scale metadata, success counts, failure counts, and validity.
        """
        return pd.DataFrame(
            [
                {
                    "scale_index": s.scale_index,
                    "scale_ratio": s.scale_ratio,
                    "n": self.n,
                    "n_prime": s.n_prime,
                    "sa": s.sa,
                    "n_bootstrap_requested": s.n_bootstrap_requested,
                    "n_success": s.n_success,
                    "n_failure": len(s.fit_failures),
                    "is_valid": s.is_valid,
                }
                for s in self.scales
            ]
        )


class MultiScaleBootstrap:
    """Run a multi-scale (varying resample size) bootstrap for any LiNGAM
    model that exposes ``fit(X)`` and ``adjacency_matrix_``.

    Parameters
    ----------
    model : object
        An unfitted model instance (e.g. ``lingam.ABICLiNGAM()`` or
        ``lingam.DirectLiNGAM()``). A fresh copy is made per replicate via
        :meth:`_make_model` (default: ``copy.deepcopy``), so the instance
        passed in is never mutated.
    scale_ratios : tuple of float
        Ratios ``n_prime / n`` used to build the scale grid. The scale
        passed to scaleboot is ``sa = n / n_prime`` computed from the
        *actual* (rounded) ``n_prime``, not from ``1 / scale_ratio``.
    n_bootstrap : int
        Number of bootstrap replicates requested per scale. The number of
        *successful* replicates (``n_success``) may be lower and is what is
        actually reported to scaleboot as ``nb``.
    random_state : int
        Master seed for resampling. Model-internal randomness (e.g. ABIC's
        own ``seed``) is the caller's responsibility and is not touched
        here.
    min_success_rate : float
        Minimum fraction of successful fits required for a scale to be
        considered valid. Scales below this rate are excluded from
        scaleboot fitting (but are still reported for diagnostics).

    Notes
    -----
    Override :meth:`_make_model`, :meth:`_resample`,
    :meth:`_fit_model`, or :meth:`_extract_matrices` in a subclass to adapt
    to models with a non-standard interface. Exceptions from these hooks are
    recorded as failed replicates. No other change to this class is required.

    Returned results retain raw adjacency matrices. Edge presence is decided
    by hypothesis extraction with ``min_causal_effect``, so changing that
    threshold does not require another bootstrap run.

    There is no built-in run persistence/resume: callers who need to
    resume a long-running experiment should persist the returned
    ``MultiScaleBootstrapResult`` themselves (e.g. via ``pickle``).
    """

    def __init__(
        self,
        model,
        scale_ratios=(0.5, 0.7, 1.0, 1.4, 2.0),
        n_bootstrap=20,
        random_state=0,
        min_success_rate=0.5,
    ):
        """Construct a multi-scale bootstrap runner.

        Parameters
        ----------
        model : object
            Unfitted LiNGAM-compatible model.
        scale_ratios : sequence of float, optional
            Resampled-size ratios ``n_prime / n``.
        n_bootstrap : int, optional
            Number of replicates requested per scale.
        random_state : int or None, optional
            Master seed for deterministic resampling.
        min_success_rate : float, optional
            Minimum success fraction required for a valid scale.
        """
        scale_ratios = tuple(scale_ratios)
        if len(scale_ratios) == 0:
            raise ValueError("scale_ratios must not be empty.")
        if any(r <= 0 for r in scale_ratios):
            raise ValueError("scale_ratios must all be positive.")
        if (
            isinstance(n_bootstrap, bool)
            or not isinstance(n_bootstrap, (int, np.integer))
            or n_bootstrap < 1
        ):
            raise ValueError("n_bootstrap must be a positive integer.")
        if not (0.0 <= min_success_rate <= 1.0):
            raise ValueError("min_success_rate must be in [0, 1].")

        self.model = model
        self.scale_ratios = scale_ratios
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.min_success_rate = min_success_rate

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------
    def _make_model(self):
        """Return a fresh, unfitted model instance for one replicate."""
        return copy.deepcopy(self.model)

    def _resample(self, X, n_prime, rng):
        """Draw ``n_prime`` rows from ``X`` with replacement."""
        n = X.shape[0]
        idx = rng.integers(0, n, size=n_prime)
        return X[idx]

    def _fit_model(self, model, X):
        """Fit one model.

        Parameters
        ----------
        model : object
            Fresh model instance for the current replicate.
        X : ndarray, shape (n_samples, n_features)
            Resampled observations.

        Notes
        -----
        Override this hook when a model needs parameters prepared before
        calling its ``fit`` method.
        """
        model.fit(X)

    def _extract_matrices(self, model):
        """Extract the adjacency matrix from a fitted model.

        Parameters
        ----------
        model : object
            Fitted model instance.

        Returns
        -------
        ndarray, shape (n_features, n_features)
            Estimated adjacency matrix using the LiNGAM convention
            ``adjacency[to, from]``.

        Notes
        -----
        Override this hook for models that expose adjacency matrices under a
        different attribute or return more than one matrix.
        """
        return np.asarray(model.adjacency_matrix_, dtype=float)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fit_once(self, X):
        """Fit one replicate and extract its adjacency matrix.

        Exceptions from model creation, fitting, or matrix extraction are
        converted into a failed :class:`FitOutcome` so the bootstrap can
        continue with the remaining replicates.
        """
        try:
            model = self._make_model()
            self._fit_model(model, X)
            adjacency = self._extract_matrices(model)
        except Exception as exc:  # noqa: BLE001 - failures are recorded, not silenced
            return FitOutcome(
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        return FitOutcome(adjacency=adjacency)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def bootstrap(self, X, variable_names=None):
        """Run the full multi-scale bootstrap experiment.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        variable_names : sequence of str, optional
            Defaults to ``["x0", "x1", ...]``.

        Returns
        -------
        MultiScaleBootstrapResult
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-dimensional array; got shape {X.shape}."
            )
        n, d = X.shape

        if variable_names is None:
            variable_names = [f"x{i}" for i in range(d)]
        else:
            variable_names = list(variable_names)
            if len(variable_names) != d:
                raise ValueError(
                    f"variable_names has {len(variable_names)} entries, but X "
                    f"has {d} columns."
                )

        seed_seq = np.random.SeedSequence(self.random_state)
        scale_seeds = seed_seq.spawn(len(self.scale_ratios))

        scales = []
        for scale_index, (scale_ratio, scale_seed) in enumerate(
            zip(self.scale_ratios, scale_seeds)
        ):
            n_prime, sa = _resolve_scale(n, scale_ratio, d)
            rep_seeds = scale_seed.spawn(self.n_bootstrap)

            adjacency_list = []
            fit_failures = []
            for rep_index, rep_seed in enumerate(rep_seeds):
                rng = np.random.default_rng(rep_seed)
                X_resampled = self._resample(X, n_prime, rng)
                outcome = self._fit_once(X_resampled)

                if not outcome.ok:
                    fit_failures.append(
                        FitFailure(
                            replicate_index=rep_index,
                            error_type=outcome.error_type or "UnknownError",
                            error_message=outcome.error_message or "",
                        )
                    )
                    continue

                adjacency_list.append(outcome.adjacency)

            n_success = len(adjacency_list)
            if n_success > 0:
                adjacency_matrices = np.stack(adjacency_list, axis=0)
            else:
                adjacency_matrices = np.empty((0, d, d))

            scale_result = SingleScaleBootstrapResult(
                scale_index=scale_index,
                scale_ratio=scale_ratio,
                n_prime=n_prime,
                sa=sa,
                n_bootstrap_requested=self.n_bootstrap,
                n_success=n_success,
                adjacency_matrices=adjacency_matrices,
                fit_failures=fit_failures,
                min_success_rate=self.min_success_rate,
            )
            scales.append(scale_result)

        n_valid = sum(1 for s in scales if s.is_valid)
        if n_valid < _MIN_VALID_SCALES:
            raise ValueError(
                f"Only {n_valid} of {len(scales)} scales met "
                f"min_success_rate={self.min_success_rate}; need at least "
                f"{_MIN_VALID_SCALES}. Consider lowering min_success_rate, "
                "increasing n_bootstrap, or adjusting scale_ratios."
            )

        return MultiScaleBootstrapResult(
            scales=scales,
            n=n,
            variable_names=variable_names,
        )
