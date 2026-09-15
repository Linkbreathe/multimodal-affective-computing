"""Compatibility shim: ``real_time_ml.config.layers`` is now ``mac.config.layers``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.config.layers as _target

sys.modules[__name__] = _target
