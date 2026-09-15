"""Compatibility shim: ``src.adaptive.metrics`` is now ``mac.adaptive.offline.metrics``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.adaptive.offline.metrics as _target

sys.modules[__name__] = _target
