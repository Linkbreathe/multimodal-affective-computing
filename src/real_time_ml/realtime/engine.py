"""Compatibility shim: ``real_time_ml.realtime.engine`` is now ``mac.realtime.engine``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.realtime.engine as _target

sys.modules[__name__] = _target
