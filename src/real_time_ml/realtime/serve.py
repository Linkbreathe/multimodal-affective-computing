"""Compatibility shim: ``real_time_ml.realtime.serve`` is now ``mac.realtime.serve``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.realtime.serve as _target

sys.modules[__name__] = _target
