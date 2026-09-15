"""Compatibility shim: ``src.adaptive.replay`` is now ``mac.adaptive.offline.replay``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.adaptive.offline.replay as _target

sys.modules[__name__] = _target
