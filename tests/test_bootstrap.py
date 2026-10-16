import numpy as np

from pyscaleboot import MultiScaleBootstrap


class ZeroModel:
    def fit(self, X):
        self.adjacency_matrix_ = np.zeros((X.shape[1], X.shape[1]))


def test_scale_sizes_and_sa():
    result = MultiScaleBootstrap(
        ZeroModel(), scale_ratios=(0.5, 1.0, 2.0), n_bootstrap=3
    ).bootstrap(np.ones((10, 2)))

    assert [s.n_prime for s in result.scales] == [5, 10, 20]
    assert [s.sa for s in result.scales] == [2.0, 1.0, 0.5]
    for s in result.scales:
        assert s.adjacency_matrices_.shape == (3, 2, 2)
        assert s.n_success == 3
        assert s.n_failure == 0
        assert s.failures == []


def test_variable_names_default_and_custom():
    msb = MultiScaleBootstrap(ZeroModel(), scale_ratios=(1.0,), n_bootstrap=1)
    assert msb.bootstrap(np.ones((4, 2))).variable_names == ["x0", "x1"]
    assert msb.bootstrap(np.ones((4, 2)), ["a", "b"]).variable_names == ["a", "b"]


def test_same_random_state_gives_same_resamples():
    class RecordingModel:
        def fit(self, X):
            self.adjacency_matrix_ = np.full((2, 2), X.sum())

    X = np.random.default_rng(0).normal(size=(20, 2))
    results = [
        MultiScaleBootstrap(
            RecordingModel(), scale_ratios=(1.0,), n_bootstrap=5, random_state=1
        ).bootstrap(X)
        for _ in range(2)
    ]
    np.testing.assert_array_equal(
        results[0].scales[0].adjacency_matrices_,
        results[1].scales[0].adjacency_matrices_,
    )


def test_failure_does_not_stop_current_or_subsequent_scales():
    calls = []

    class FailsOnSecondFit:
        def fit(self, X):
            calls.append(X.shape[0])
            if len(calls) == 2:
                raise RuntimeError("synthetic failure")
            self.adjacency_matrix_ = np.zeros((X.shape[1], X.shape[1]))

    result = MultiScaleBootstrap(
        FailsOnSecondFit(), scale_ratios=(0.5, 1.0), n_bootstrap=3
    ).bootstrap(np.ones((4, 2)))

    assert calls == [2, 2, 2, 4, 4, 4]
    failed, complete = result.scales
    assert failed.adjacency_matrices_.shape == (2, 2, 2)
    assert failed.n_success == 2
    assert failed.n_failure == 1
    assert failed.failures == [(1, "RuntimeError: synthetic failure")]
    assert complete.n_success == 3
    assert complete.n_failure == 0
    assert complete.failures == []

    summary = result.scale_summary()
    assert list(summary.columns) == [
        "scale_index", "scale_ratio", "n", "n_prime", "sa",
        "n_bootstrap", "n_success", "n_failure",
    ]
    assert list(summary["n_success"]) == [2, 3]
    assert list(summary["n_failure"]) == [1, 0]
    assert (summary["n_success"] + summary["n_failure"] == summary["n_bootstrap"]).all()


def test_multiple_failures_record_indices_and_only_successful_matrices():
    class IntermittentModel:
        def __init__(self):
            self.calls = 0

        def fit(self, X):
            self.calls += 1
            if self.calls in (1, 3, 4):
                raise ValueError(f"failure {self.calls}")
            self.adjacency_matrix_ = np.full((2, 2), self.calls, dtype=float)

    model = IntermittentModel()
    result = MultiScaleBootstrap(
        model, scale_ratios=(1.0,), n_bootstrap=5
    ).bootstrap(np.ones((4, 2)))

    scale = result.scales[0]
    assert model.calls == 5
    assert scale.n_success == 2
    assert scale.n_failure == 3
    assert scale.failures == [
        (0, "ValueError: failure 1"),
        (2, "ValueError: failure 3"),
        (3, "ValueError: failure 4"),
    ]
    np.testing.assert_array_equal(
        scale.adjacency_matrices_, np.stack([np.full((2, 2), 2), np.full((2, 2), 5)])
    )


def test_zero_bootstrap_scale_has_no_successes_or_failures():
    result = MultiScaleBootstrap(
        ZeroModel(), scale_ratios=(1.0,), n_bootstrap=0
    ).bootstrap(np.ones((4, 2)))

    scale = result.scales[0]
    assert scale.adjacency_matrices_.shape == (0, 2, 2)
    assert scale.n_success == 0
    assert scale.n_failure == 0
    assert scale.failures == []


def test_all_replicates_failing_gives_empty_scale():
    class AlwaysFails:
        def fit(self, X):
            raise RuntimeError("synthetic failure")

    result = MultiScaleBootstrap(
        AlwaysFails(), scale_ratios=(1.0,), n_bootstrap=2
    ).bootstrap(np.ones((4, 2)))

    scale = result.scales[0]
    assert scale.adjacency_matrices_.shape == (0, 2, 2)
    assert scale.n_success == 0
    assert scale.n_failure == 2
    assert scale.failures == [
        (0, "RuntimeError: synthetic failure"),
        (1, "RuntimeError: synthetic failure"),
    ]
