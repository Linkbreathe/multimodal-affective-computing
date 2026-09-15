"""Compatibility shim: ``real_time_ml.adaptive_control.contracts`` is now ``mac.adaptive.control.contracts``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.adaptive.control.contracts as _target

sys.modules[__name__] = _target
