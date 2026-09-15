"""Compatibility shim: ``real_time_ml.adaptive_control.physio_monitor`` is now ``mac.adaptive.control.physio_monitor``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.adaptive.control.physio_monitor as _target

sys.modules[__name__] = _target
