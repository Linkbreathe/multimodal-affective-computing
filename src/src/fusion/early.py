"""Compatibility shim: ``src.fusion.early`` is now ``mac.fusion.early``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.early as _target

sys.modules[__name__] = _target
