"""Compatibility shim: ``real_time_ml.data.index`` is now ``mac.data.index``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.index as _target

sys.modules[__name__] = _target
