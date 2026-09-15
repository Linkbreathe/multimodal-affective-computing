"""Compatibility shim: ``real_time_ml.schema`` is now ``mac.schema``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.schema as _target

sys.modules[__name__] = _target
