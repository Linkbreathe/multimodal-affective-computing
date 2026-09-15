"""Compatibility shim: ``real_time_ml.modeling.train`` is now ``mac.training.train``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.training.train as _target

sys.modules[__name__] = _target
