"""Compatibility shim: ``real_time_ml.training.state`` is now ``mac.training.state``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.training.state as _target

sys.modules[__name__] = _target
