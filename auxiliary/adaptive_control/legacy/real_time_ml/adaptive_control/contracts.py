"""Historical compatibility shim for archived Adaptive Control contracts.

Aliases the archived module so old research snapshots can be inspected.
"""

import sys

import auxiliary.adaptive_control.contracts as _target

sys.modules[__name__] = _target
