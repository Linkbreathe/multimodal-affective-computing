"""Compatibility shim: ``real_time_ml.modeling.video_train`` is now ``mac.training.video_train``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.training.video_train as _target

sys.modules[__name__] = _target
