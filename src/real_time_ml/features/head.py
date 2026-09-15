"""Compatibility shim: ``real_time_ml.features.head`` is now ``mac.features.head``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.features.head as _target

sys.modules[__name__] = _target
