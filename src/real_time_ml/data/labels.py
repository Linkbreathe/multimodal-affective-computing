"""Compatibility shim: ``real_time_ml.data.labels`` is now ``mac.data.labels``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.labels as _target

sys.modules[__name__] = _target
