"""Compatibility shim: ``real_time_ml.adaptive_control.settings`` is now ``mac.adaptive.control.settings``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.adaptive.control.settings as _target

sys.modules[__name__] = _target
