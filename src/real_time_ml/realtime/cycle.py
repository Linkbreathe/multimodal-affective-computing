"""Compatibility shim: ``real_time_ml.realtime.cycle`` is now ``mac.realtime.cycle``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.realtime.cycle as _target

sys.modules[__name__] = _target
