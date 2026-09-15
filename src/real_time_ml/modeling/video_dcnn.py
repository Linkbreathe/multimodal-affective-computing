"""Compatibility shim: ``real_time_ml.modeling.video_dcnn`` is now ``mac.models.video_dcnn``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.models.video_dcnn as _target

sys.modules[__name__] = _target
