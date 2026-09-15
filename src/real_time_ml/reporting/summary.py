"""Compatibility shim: ``real_time_ml.reporting.summary`` is now ``mac.reporting.summary``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.reporting.summary as _target

sys.modules[__name__] = _target
