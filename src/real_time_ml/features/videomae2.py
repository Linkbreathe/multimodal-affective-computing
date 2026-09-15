"""Compatibility shim: ``real_time_ml.features.videomae2`` is now ``mac.features.videomae2``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.features.videomae2 as _target

sys.modules[__name__] = _target
