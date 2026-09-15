"""Historical compatibility shim for archived Adaptive Control policy.

Aliases the archived module so old research snapshots can be inspected.
"""

import sys

import Auxiliary.adaptive_control.policy as _target

sys.modules[__name__] = _target
