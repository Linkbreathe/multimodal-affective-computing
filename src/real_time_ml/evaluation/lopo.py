"""Compatibility shim: ``real_time_ml.evaluation.lopo`` is now ``mac.evaluation.lopo``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.evaluation.lopo as _target

sys.modules[__name__] = _target
