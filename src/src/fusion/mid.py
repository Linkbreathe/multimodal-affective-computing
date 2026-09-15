"""Compatibility shim: ``src.fusion.mid`` is now ``mac.fusion.mid``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.mid as _target

sys.modules[__name__] = _target
