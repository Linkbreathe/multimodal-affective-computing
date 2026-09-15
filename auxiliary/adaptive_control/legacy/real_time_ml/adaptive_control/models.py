"""Historical compatibility shim for archived Adaptive Control models.

Aliases the archived module so old research snapshots can be inspected.
"""

import sys

import auxiliary.adaptive_control.models as _target

sys.modules[__name__] = _target
