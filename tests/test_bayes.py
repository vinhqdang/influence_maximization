import numpy as np

from im_lab.bayes import BetaBernoulliTracker


def test_posterior_mean_converges_to_true_probability():
    """Basic consistency check: repeatedly feeding Bernoulli(p_true) trials into the
    tracker should drive the posterior mean close to p_true as observations grow."""
    rng = np.random.default_rng(42)
    p_true = 0.73
    tracker = BetaBernoulliTracker(alpha0=1.0, beta0=1.0)

    key = ("p_plus", "u", "v")
    # Before any observation, mean should equal the prior mean (0.5 for Beta(1,1)).
    assert abs(tracker.mean(key) - 0.5) < 1e-9

    n_trials = 5000
    for _ in range(n_trials):
        success = rng.random() < p_true
        tracker.update(key, success)

    assert abs(tracker.mean(key) - p_true) < 0.03


def test_independent_keys_do_not_interfere():
    tracker = BetaBernoulliTracker()
    for _ in range(100):
        tracker.update(("p_plus", 0, 1), True)
    for _ in range(100):
        tracker.update(("p_minus", 0, 1), False)

    assert tracker.mean(("p_plus", 0, 1)) > 0.95
    assert tracker.mean(("p_minus", 0, 1)) < 0.05
    # An untouched key should still be at the prior mean.
    assert abs(tracker.mean(("q", 5)) - 0.5) < 1e-9


def test_variance_shrinks_with_more_observations():
    tracker = BetaBernoulliTracker()
    key = ("q", 0)
    var_before = tracker.variance(key)
    for _ in range(200):
        tracker.update(key, True)
    var_after = tracker.variance(key)
    assert var_after < var_before
