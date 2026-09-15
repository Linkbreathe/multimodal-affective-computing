"""Compatibility shim: ``real_time_ml.features.eye`` is now ``mac.features.eye``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.features.eye as _target

sys.modules[__name__] = _target
