"""Compatibility shim: ``real_time_ml.data.xlsx`` is now ``mac.data.xlsx``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.xlsx as _target

sys.modules[__name__] = _target
