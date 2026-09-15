"""Compatibility shim: ``real_time_ml.adaptive_control.models`` is now ``mac.adaptive.control.models``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.adaptive.control.models as _target

sys.modules[__name__] = _target
