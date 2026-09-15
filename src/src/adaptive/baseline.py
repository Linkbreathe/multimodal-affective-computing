"""Compatibility shim: ``src.adaptive.baseline`` is now ``mac.adaptive.offline.baseline``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.adaptive.offline.baseline as _target

sys.modules[__name__] = _target
