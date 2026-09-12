"""Per-round node actions and their costs.

Cost asymmetry (CONVERT=5, MAINTAIN=1, NONE=0) is part of the model spec: pushing an
inactive node to active is deliberately made 5x more expensive than sustaining an
already-active node against decay/backfire, mirroring the intuition that acquisition
("convert a customer") is materially harder than retention ("keep them") in the
applications this model is meant to stylize.
"""

from enum import Enum


class Action(Enum):
    NONE = "NONE"
    CONVERT = "CONVERT"
    MAINTAIN = "MAINTAIN"


# Cost of applying each action to a single node in a single round.
ACTION_COST = {
    Action.NONE: 0,
    Action.MAINTAIN: 1,
    Action.CONVERT: 5,
}


def is_valid_action(action: Action, is_active: bool) -> bool:
    """CONVERT is only meaningful on an inactive node; MAINTAIN only on an active one.

    NONE is always valid. Applying CONVERT to an already-active node (or MAINTAIN to
    an inactive one) is not meaningful under the model and is treated as an error by
    callers rather than silently downgraded to NONE, to keep policy bugs visible.
    """
    if action == Action.NONE:
        return True
    if action == Action.CONVERT:
        return not is_active
    if action == Action.MAINTAIN:
        return is_active
    raise ValueError(f"Unknown action: {action}")
