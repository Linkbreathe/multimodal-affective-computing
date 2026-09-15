"""Compatibility shim: ``real_time_ml.features.egocentric`` is now ``mac.features.egocentric``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.features.egocentric as _target

sys.modules[__name__] = _target
