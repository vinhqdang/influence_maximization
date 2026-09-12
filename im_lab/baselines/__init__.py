"""Classical one-shot IC seed-selection baselines (plain IC: beta=0, q=0), used as
ablations against MF-BWI-Fair. Both are converted into "CONVERT the seeds at round
0, then NONE forever" sequential policies via simulator.seed_then_none_actions so
they can be compared inside the same non-progressive/backfire-capable simulator.
"""
