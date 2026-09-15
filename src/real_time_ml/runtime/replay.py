"""Compatibility shim: ``real_time_ml.runtime.replay`` is now ``mac.runtime.replay``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.runtime.replay as _target

sys.modules[__name__] = _target
