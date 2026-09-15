"""Compatibility shim: ``real_time_ml.data.io`` is now ``mac.data.io``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.io as _target

sys.modules[__name__] = _target
