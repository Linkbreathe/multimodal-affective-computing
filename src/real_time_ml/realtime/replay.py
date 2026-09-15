"""Compatibility shim: ``real_time_ml.realtime.replay`` is now ``mac.realtime.replay``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.realtime.replay as _target

sys.modules[__name__] = _target
