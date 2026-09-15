"""Historical compatibility shim for archived Adaptive Control monitoring.

Aliases the archived module so old research snapshots can be inspected.
"""

import sys

import Auxiliary.adaptive_control.physio_monitor as _target

sys.modules[__name__] = _target
