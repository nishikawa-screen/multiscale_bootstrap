import numpy as np
import pytest
from typing import Any
from lingam import DirectLiNGAM

from multiscale_bootstrap import (
    MultiScaleBootstrap,
    MultiScaleBootstrapResult,
    SingleScaleBootstrapResult,
)
from multiscale_bootstrap._bootstrap import (
    FitFailure,
    FitOutcome,
    _min_n_prime,
    _resolve_scale,
)

from ._helpers import make_chain_data


def _small_msb(model, **overrides):
    kwargs = dict(
        scale_ratios=(0.7, 1.0, 1.4),
        n_bootstrap=5,
        random_state=0,
    )
    kwargs.update(overrides)
    return MultiScaleBootstrap(model, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scale_ratios=()),
        dict(scale_ratios=(0.5, -1.0)),
        dict(n_bootstrap=0),
        dict(n_bootstrap=1.5),
        dict(n_bootstrap=True),
        dict(min_success_rate=1.5),
    ],
)
def test_invalid_constructor_args_raise_value_error(kwargs: dict):
    with pytest.raises(ValueError):
        MultiScaleBootstrap(DirectLiNGAM(), **kwargs)


def test_constructor_rejects_removed_keyword():
    removed_keyword = "_".join(("edge", "threshold"))
    with pytest.raises(TypeError):
        MultiScaleBootstrap(DirectLiNGAM(), **{removed_keyword: 0.1})


def test_resolve_scale_computes_n_prime_and_sa():
    n_prime, sa = _resolve_scale(n=200, scale_ratio=0.5, d=3)
    assert n_prime == 100
    assert sa == pytest.approx(200 / 100)


def test_resolve_scale_raises_below_min_n_prime():
    with pytest.raises(ValueError):
        _resolve_scale(n=20, scale_ratio=0.5, d=5)


def test_min_n_prime_floor():
    assert _min_n_prime(d=3) == 30
    assert _min_n_prime(d=20) == 41


def test_bootstrap_runs_with_direct_lingam():
    X = make_chain_data(n=300, seed=0)
    msb = _small_msb(DirectLiNGAM())
    result = msb.bootstrap(X, variable_names=["x0", "x1", "x2"])

    assert result.n == 300
    assert len(result.variable_names) == 3
    assert len(result.scales) == 3
    removed_attribute = "_".join(("edge", "states"))

    for scale in result.scales:
        assert scale.n_success + len(scale.fit_failures) == scale.n_bootstrap_requested
        assert scale.adjacency_matrices.shape == (scale.n_success, 3, 3)
        assert not hasattr(scale, removed_attribute)
        assert scale.is_valid


class _RawAdjacencyModel:
    def fit(self, X):
        self.adjacency_matrix_ = np.array([[0.0, np.nan], [-0.05, 0.0]])


def test_bootstrap_preserves_raw_adjacency_matrices():
    result = MultiScaleBootstrap(
        _RawAdjacencyModel(), scale_ratios=(0.7, 1.0), n_bootstrap=2
    ).bootstrap(np.zeros((100, 2)))
    expected = np.array([[0.0, np.nan], [-0.05, 0.0]])
    removed_attribute = "_".join(("edge", "states"))

    for scale in result.scales:
        assert scale.n_success == scale.adjacency_matrices.shape[0]
        np.testing.assert_allclose(
            scale.adjacency_matrices,
            np.repeat(expected[np.newaxis, :, :], scale.n_success, axis=0),
            equal_nan=True,
        )
        assert not hasattr(scale, removed_attribute)


class _PreFitModel:
    def fit(self, X):
        if not hasattr(self, "prepared_n"):
            raise RuntimeError("model was not prepared")
        n_features = X.shape[1]
        self.adjacency_matrix_ = np.zeros((n_features, n_features))


class _PreFitBootstrap(MultiScaleBootstrap):
    def _fit_model(self, model, X):
        model.prepared_n = X.shape[0]
        model.fit(X)


_make_failure_counter = {"n": 0}


class _MakeFailureBootstrap(_PreFitBootstrap):
    def _make_model(self):
        _make_failure_counter["n"] += 1
        if _make_failure_counter["n"] == 1:
            raise RuntimeError("synthetic model creation failure")
        return super()._make_model()


def test_model_creation_failures_are_recorded_and_other_replicates_continue():
    _make_failure_counter["n"] = 0
    X = make_chain_data(n=100, seed=5)
    result = _MakeFailureBootstrap(
        _PreFitModel(), scale_ratios=(0.7, 1.0), n_bootstrap=3
    ).bootstrap(X)

    first_scale = result.scales[0]
    assert first_scale.n_success == 2
    assert len(first_scale.fit_failures) == 1
    assert first_scale.fit_failures[0].error_type == "RuntimeError"
    assert first_scale.adjacency_matrices.shape[0] == first_scale.n_success


def test_fit_model_hook_prepares_model_before_fit():
    X = make_chain_data(n=100, seed=5)
    msb = _PreFitBootstrap(
        _PreFitModel(), scale_ratios=(0.7, 1.0), n_bootstrap=2
    )

    result = msb.bootstrap(X)

    assert all(scale.n_success == 2 for scale in result.scales)


_extract_failure_counter = {"n": 0}


class _ExtractFailureBootstrap(MultiScaleBootstrap):
    def _fit_model(self, model, X):
        model.prepared_n = X.shape[0]
        model.fit(X)

    def _extract_matrices(self, model):
        _extract_failure_counter["n"] += 1
        if _extract_failure_counter["n"] == 1:
            raise LookupError("synthetic extraction failure")
        return super()._extract_matrices(model)


def test_extraction_failures_are_recorded_and_other_replicates_continue():
    _extract_failure_counter["n"] = 0
    X = make_chain_data(n=100, seed=6)
    msb = _ExtractFailureBootstrap(
        _PreFitModel(), scale_ratios=(0.7, 1.0), n_bootstrap=3
    )

    result = msb.bootstrap(X)

    first_scale = result.scales[0]
    assert first_scale.n_success == 2
    assert len(first_scale.fit_failures) == 1
    assert first_scale.fit_failures[0].error_type == "LookupError"
    assert first_scale.adjacency_matrices.shape[0] == first_scale.n_success
    assert all(
        scale.n_success + len(scale.fit_failures) == scale.n_bootstrap_requested
        for scale in result.scales
    )


def test_bootstrap_reproducible_with_same_seed():
    X = make_chain_data(n=200, seed=1)
    result_a = _small_msb(DirectLiNGAM(), n_bootstrap=4).bootstrap(X)
    result_b = _small_msb(DirectLiNGAM(), n_bootstrap=4).bootstrap(X)

    for sa, sb in zip(result_a.scales, result_b.scales):
        np.testing.assert_array_equal(sa.adjacency_matrices, sb.adjacency_matrices)


def test_default_variable_names():
    X = make_chain_data(n=150, seed=2)
    result = _small_msb(DirectLiNGAM(), n_bootstrap=3).bootstrap(X)
    assert result.variable_names == ["x0", "x1", "x2"]


def test_variable_names_length_mismatch_raises():
    X = make_chain_data(n=150, seed=2)
    msb = _small_msb(DirectLiNGAM(), n_bootstrap=3)
    with pytest.raises(ValueError):
        msb.bootstrap(X, variable_names=["only_one"])


class _AlwaysFailModel:
    """Dummy model whose fit() always raises, to exercise failure handling."""

    def fit(self, X):
        raise ValueError("synthetic failure")


def test_all_failures_raises_due_to_min_valid_scales():
    X = make_chain_data(n=100, seed=3)
    msb = MultiScaleBootstrap(
        _AlwaysFailModel(),
        scale_ratios=(0.7, 1.0, 1.4),
        n_bootstrap=3,
        min_success_rate=0.1,
    )
    with pytest.raises(ValueError):
        msb.bootstrap(X)


# Module-level (not instance-level) counter: MultiScaleBootstrap deep-copies
# the model for every fit call, so an instance attribute counter would be
# reset every time and never observe failures.
_flaky_call_counter = {"n": 0}


class _FlakyModel:
    """Fails on even calls (deterministic given call order)."""

    def fit(self, X):
        _flaky_call_counter["n"] += 1
        if _flaky_call_counter["n"] % 2 == 0:
            raise ValueError("synthetic even-call failure")
        n_features = X.shape[1]
        self.adjacency_matrix_ = np.zeros((n_features, n_features))
        self.causal_order_ = list(range(n_features))


def test_partial_failures_are_recorded():
    _flaky_call_counter["n"] = 0
    X = make_chain_data(n=100, seed=4)
    msb = MultiScaleBootstrap(
        _FlakyModel(),
        # 2 scales, to satisfy the fixed minimum-valid-scales floor.
        scale_ratios=(0.7, 1.0),
        n_bootstrap=6,
        min_success_rate=0.1,
    )
    result = msb.bootstrap(X)
    scale = result.scales[0]
    assert len(scale.fit_failures) > 0
    assert scale.n_success + len(scale.fit_failures) == scale.n_bootstrap_requested
    assert all(ff.error_type == "ValueError" for ff in scale.fit_failures)


def _make_scale(**overrides) -> SingleScaleBootstrapResult:
    kwargs: dict[str, Any] = dict(
        scale_index=0,
        scale_ratio=1.0,
        n_prime=100,
        sa=1.0,
        n_bootstrap_requested=10,
        n_success=10,
        adjacency_matrices=np.zeros((10, 2, 2)),
        fit_failures=[],
        min_success_rate=0.5,
    )
    kwargs.update(overrides)
    return SingleScaleBootstrapResult(**kwargs)


def test_failure_count_is_derived_from_fit_failures():
    failures = [
        FitFailure(replicate_index=i, error_type="ValueError", error_message="x")
        for i in range(3)
    ]
    scale = _make_scale(n_success=7, fit_failures=failures)
    assert len(scale.fit_failures) == 3
    assert scale.n_success + len(scale.fit_failures) == scale.n_bootstrap_requested


def test_fit_outcome_ok_matches_adjacency_presence():
    ok_outcome = FitOutcome(adjacency=np.zeros((2, 2)))
    failed_outcome = FitOutcome(error_type="ValueError", error_message="boom")
    assert ok_outcome.ok is True
    assert failed_outcome.ok is False


def test_multi_scale_result_variable_names_are_preserved():
    result = MultiScaleBootstrapResult(
        scales=[_make_scale()], n=100, variable_names=["x0", "x1", "x2"]
    )
    assert len(result.variable_names) == 3


@pytest.mark.parametrize(
    "n_success, n_bootstrap_requested, min_success_rate, expected",
    [
        (4, 10, 0.5, False),  # below threshold
        (5, 10, 0.5, True),  # exactly at threshold
        (6, 10, 0.5, True),  # above threshold
    ],
)
def test_is_valid_threshold_boundaries(
    n_success, n_bootstrap_requested, min_success_rate, expected
):
    scale = _make_scale(
        n_success=n_success,
        n_bootstrap_requested=n_bootstrap_requested,
        min_success_rate=min_success_rate,
    )
    assert scale.is_valid is expected


def test_is_valid_no_zero_division_when_n_bootstrap_requested_is_zero():
    scale = _make_scale(n_success=0, n_bootstrap_requested=0)
    assert scale.is_valid is False


def test_scale_n_prime_uses_parent_result_n():
    scale = _make_scale(n_prime=70)
    result = MultiScaleBootstrapResult(scales=[scale], n=100, variable_names=["x0", "x1"])
    assert result.n == 100
    assert not hasattr(scale, "n")
