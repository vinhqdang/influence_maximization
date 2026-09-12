"""im_lab: classical-algorithms toolkit for a generalized influence-maximization variant.

Relaxes four assumptions of Kempe-Kleinberg-Tardos (2003) independent-cascade IM
simultaneously: group fairness constraints, non-progressive (recoverable) dynamics,
non-monotone "backfire" influence, and Bayesian-uncertain edge/node parameters.

No machine learning / reinforcement learning / neural networks are used anywhere in
this package; all algorithms are classical (combinatorial greedy, Monte Carlo
estimation, conjugate Bayesian updating, mean-field fixed points).
"""

__all__ = [
    "actions",
    "graphs",
    "bayes",
    "simulator",
    "fairness",
    "lagrangian_index",
    "mf_bwi_fair",
]
