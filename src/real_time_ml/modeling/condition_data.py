"""Compatibility shim: ``real_time_ml.modeling.condition_data`` is now ``mac.data.condition_data``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.condition_data as _target

sys.modules[__name__] = _target
