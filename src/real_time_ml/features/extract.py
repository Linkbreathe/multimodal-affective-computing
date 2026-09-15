"""Compatibility shim: ``real_time_ml.features.extract`` is now ``mac.features.extract``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.features.extract as _target

sys.modules[__name__] = _target
