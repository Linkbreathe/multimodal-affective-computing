"""Compatibility shim: ``src.adaptive.condition_grid`` is now ``mac.adaptive.offline.condition_grid``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.adaptive.offline.condition_grid as _target

sys.modules[__name__] = _target
