"""Compatibility shim: ``real_time_ml.data.video`` is now ``mac.data.video``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.video as _target

sys.modules[__name__] = _target
