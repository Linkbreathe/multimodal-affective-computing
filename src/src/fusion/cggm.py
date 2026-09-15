"""Compatibility shim: ``src.fusion.cggm`` is now ``mac.fusion.cggm``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.cggm as _target

sys.modules[__name__] = _target
