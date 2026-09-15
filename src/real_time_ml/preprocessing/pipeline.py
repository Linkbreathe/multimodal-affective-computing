"""Compatibility shim: ``real_time_ml.preprocessing.pipeline`` is now ``mac.preprocessing.pipeline``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.preprocessing.pipeline as _target

sys.modules[__name__] = _target
