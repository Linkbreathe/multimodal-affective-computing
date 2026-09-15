"""Compatibility shim: ``real_time_ml.data.alignment`` is now ``mac.data.alignment``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.alignment as _target

sys.modules[__name__] = _target
