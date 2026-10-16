"""Core multiscale bootstrap engine for LiNGAM-compatible models."""

import numpy as np
import pandas as pd
from sklearn.utils import check_array


class SingleScaleBootstrapResult:
    """Bootstrap outcome for one scale (one value of ``n_prime`` / ``sa``).

    Parameters
    ----------
    scale_ratio : float
        Requested ratio ``n_prime / n``.
    n_prime : int
        Actual resampled sample size.
    sa : float
        ``n / n_prime`` for the actual sample size.
    n_bootstrap : int
        Number of requested replicates.
    adjacency_matrices : ndarray, shape (n_success, d, d)
        Raw adjacency matrices of the successful replicates (``NaN`` marks a
        bidirected edge).
    failures : list of (int, str)
        ``(replicate_index, "ExceptionType: message")`` of each failed replicate.
    """

    def __init__(
        self, scale_ratio, n_prime, sa, n_bootstrap, adjacency_matrices, failures=()
    ):
        self.scale_ratio = scale_ratio
        self.n_prime = n_prime
        self.sa = sa
        self.n_bootstrap = n_bootstrap
        self.adjacency_matrices_ = adjacency_matrices
        self.failures = list(failures)

    @property
    def n_success(self):
        """Number of successful replicates."""
        return len(self.adjacency_matrices_)

    @property
    def n_failure(self):
        """Number of failed replicates."""
        return len(self.failures)


class MultiScaleBootstrapResult:
    """Aggregate result across all scales.

    The result holds raw adjacency matrices before thresholding. Hypothesis
    extraction applies ``min_causal_effect``, so the same result can be
    re-aggregated with different thresholds without rerunning the bootstrap.

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
        self.scales = scales
        self.n = n
        self.variable_names = variable_names

    def scale_summary(self):
        """Return a summary table containing one row for each scale.

        Returns
        -------
        pandas.DataFrame
            Scale sizes and the success and failure counts.
        """
        return pd.DataFrame(
            [
                {
                    "scale_index": i,
                    "scale_ratio": s.scale_ratio,
                    "n": self.n,
                    "n_prime": s.n_prime,
                    "sa": s.sa,
                    "n_bootstrap": s.n_bootstrap,
                    "n_success": s.n_success,
                    "n_failure": s.n_failure,
                }
                for i, s in enumerate(self.scales)
            ]
        )


class MultiScaleBootstrap:
    """Run a multi-scale (varying resample size) bootstrap for any LiNGAM
    model that exposes ``fit(X)`` and ``adjacency_matrix_``.

    Parameters
    ----------
    model : object
        An unfitted model instance (e.g. ``lingam.ABICLiNGAM()`` or
        ``lingam.DirectLiNGAM()``). The same instance is refitted for every
        replicate.
    scale_ratios : tuple of float
        Ratios ``n_prime / n`` used to build the scale grid. The scale
        passed to scaleboot is ``sa = n / n_prime`` computed from the
        *actual* (rounded) ``n_prime``.
    n_bootstrap : int
        Number of bootstrap replicates requested per scale. A failed fit is
        recorded in ``failures`` and the remaining replicates still run.
        Bootstrap probabilities use only the successful replicates.
    random_state : int
        Seed for resampling. Model-internal randomness (e.g. ABIC's own
        ``seed``) is the caller's responsibility.
    """

    def __init__(
        self,
        model,
        scale_ratios=(0.5, 0.7, 1.0, 1.4, 2.0),
        n_bootstrap=20,
        random_state=0,
    ):
        self.model = model
        self.scale_ratios = tuple(scale_ratios)
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def bootstrap(self, X, variable_names=None):
        """Run the full multi-scale bootstrap experiment.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        variable_names : sequence of str, optional
            Defaults to ``["x0", "x1", ...]``.

        Returns
        -------
        MultiScaleBootstrapResult
        """
        X = check_array(X)
        n, d = X.shape
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(d)]

        rng = np.random.default_rng(self.random_state)

        scales = []
        for scale_ratio in self.scale_ratios:
            n_prime = int(round(scale_ratio * n))
            adjacency_matrices = []
            failures = []
            for b in range(self.n_bootstrap):
                resampled_X = X[rng.integers(0, n, size=n_prime)]
                try:
                    self.model.fit(resampled_X)
                    adjacency_matrices.append(
                        np.array(self.model.adjacency_matrix_, dtype=float)
                    )
                except Exception as exc:
                    failures.append((b, f"{type(exc).__name__}: {exc}"))

            scales.append(
                SingleScaleBootstrapResult(
                    scale_ratio=scale_ratio,
                    n_prime=n_prime,
                    sa=n / n_prime,
                    n_bootstrap=self.n_bootstrap,
                    adjacency_matrices=np.array(adjacency_matrices).reshape(-1, d, d),
                    failures=failures,
                )
            )

        return MultiScaleBootstrapResult(
            scales=scales, n=n, variable_names=list(variable_names)
        )
