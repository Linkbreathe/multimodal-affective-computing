"""Compatibility shim: ``src.fusion.base`` is now ``mac.fusion.base``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.base as _target

sys.modules[__name__] = _target
