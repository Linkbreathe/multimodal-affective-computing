"""Compatibility shim: ``real_time_ml.training.dcnn`` is now ``mac.training.dcnn``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.training.dcnn as _target

sys.modules[__name__] = _target
