"""Compatibility shim: ``real_time_ml.utils`` is now ``mac.utils``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.utils as _target

sys.modules[__name__] = _target
