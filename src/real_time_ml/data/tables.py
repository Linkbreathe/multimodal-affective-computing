"""Compatibility shim: ``real_time_ml.data.tables`` is now ``mac.data.tables``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.tables as _target

sys.modules[__name__] = _target
