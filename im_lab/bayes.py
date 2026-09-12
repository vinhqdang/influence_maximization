"""Beta-Bernoulli conjugate posterior tracking for the uncertain edge/node parameters.

p_plus[u,v], p_minus[u,v], and q[v] are each treated as independent Bernoulli success
probabilities with independent Beta(alpha0, beta0) priors (default Beta(1,1), i.e.
uniform). Every realized activation/deactivation/recovery trial reported by the
simulator is a Bernoulli(success/failure) observation of exactly one of these
parameters, and the conjugate update is the standard alpha += 1 (success) /
beta += 1 (failure). The algorithm only ever reads posterior means; it never sees the
hidden ground-truth parameters used by the simulator.
"""

from __future__ import annotations


class BetaBernoulliTracker:
    """Independent Beta-Bernoulli posteriors keyed by an arbitrary hashable key.

    Keys are typically ('p_plus', u, v), ('p_minus', u, v), or ('q', v). Posteriors
    are created lazily on first access/update, seeded at (alpha0, beta0).
    """

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0):
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError("alpha0 and beta0 must be positive")
        self.alpha0 = alpha0
        self.beta0 = beta0
        self._alpha: dict = {}
        self._beta: dict = {}

    def _ensure(self, key) -> None:
        if key not in self._alpha:
            self._alpha[key] = self.alpha0
            self._beta[key] = self.beta0

    def update(self, key, success: bool) -> None:
        """Conjugate Beta-Bernoulli update from one observed trial outcome."""
        self._ensure(key)
        if success:
            self._alpha[key] += 1.0
        else:
            self._beta[key] += 1.0

    def mean(self, key) -> float:
        """Posterior mean alpha / (alpha + beta); returns the prior mean if unseen."""
        self._ensure(key)
        a, b = self._alpha[key], self._beta[key]
        return a / (a + b)

    def variance(self, key) -> float:
        self._ensure(key)
        a, b = self._alpha[key], self._beta[key]
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def n_obs(self, key) -> int:
        """Number of trials observed for this key (posterior concentration proxy)."""
        self._ensure(key)
        return int(round((self._alpha[key] - self.alpha0) + (self._beta[key] - self.beta0)))

    def params(self, key) -> tuple[float, float]:
        self._ensure(key)
        return self._alpha[key], self._beta[key]
