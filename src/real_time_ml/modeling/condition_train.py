"""Compatibility shim: ``real_time_ml.modeling.condition_train`` is now ``mac.training.condition_train``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.training.condition_train as _target

sys.modules[__name__] = _target
