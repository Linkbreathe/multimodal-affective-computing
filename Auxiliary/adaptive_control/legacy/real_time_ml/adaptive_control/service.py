"""Historical compatibility shim for the archived Adaptive Control service.

Aliases the archived module so old research snapshots can be inspected.
"""

import sys

import Auxiliary.adaptive_control.service as _target

sys.modules[__name__] = _target
