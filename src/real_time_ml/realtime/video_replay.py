"""Compatibility shim: ``real_time_ml.realtime.video_replay`` is now ``mac.realtime.video_replay``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.realtime.video_replay as _target

sys.modules[__name__] = _target
