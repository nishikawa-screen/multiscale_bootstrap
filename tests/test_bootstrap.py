import numpy as np

from multiscale_bootstrap import MultiScaleBootstrap


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
        assert s.is_valid
        assert s.error is None


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


def test_failure_stops_only_current_scale():
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

    assert calls == [2, 2, 4, 4, 4]
    failed, complete = result.scales
    assert failed.n_success == 1
    assert not failed.is_valid
    assert failed.error == "RuntimeError: synthetic failure"
    assert complete.is_valid

    summary = result.scale_summary()
    assert list(summary["is_valid"]) == [False, True]
    assert list(summary["n_success"]) == [1, 3]
    assert summary.loc[0, "error"] == "RuntimeError: synthetic failure"


def test_zero_bootstrap_scale_is_not_valid():
    result = MultiScaleBootstrap(
        ZeroModel(), scale_ratios=(1.0,), n_bootstrap=0
    ).bootstrap(np.ones((4, 2)))

    assert result.scales[0].n_success == 0
    assert not result.scales[0].is_valid


def test_all_replicates_failing_gives_empty_scale():
    class AlwaysFails:
        def fit(self, X):
            raise RuntimeError("synthetic failure")

    result = MultiScaleBootstrap(
        AlwaysFails(), scale_ratios=(1.0,), n_bootstrap=2
    ).bootstrap(np.ones((4, 2)))

    scale = result.scales[0]
    assert scale.adjacency_matrices_.shape == (0, 2, 2)
    assert not scale.is_valid
