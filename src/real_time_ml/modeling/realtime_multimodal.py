"""Compatibility shim: ``real_time_ml.modeling.realtime_multimodal`` is now ``mac.models.realtime_multimodal``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.models.realtime_multimodal as _target

sys.modules[__name__] = _target
