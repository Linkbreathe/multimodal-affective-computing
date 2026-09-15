"""Compatibility shim: ``real_time_ml.training.policy`` is now ``mac.training.policy``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.training.policy as _target

sys.modules[__name__] = _target
