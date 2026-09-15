"""Compatibility shim: ``real_time_ml.adaptive_control.policy`` is now ``mac.adaptive.control.policy``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.adaptive.control.policy as _target

sys.modules[__name__] = _target
