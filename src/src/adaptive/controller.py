"""Compatibility shim: ``src.adaptive.controller`` is now ``mac.adaptive.offline.controller``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.adaptive.offline.controller as _target

sys.modules[__name__] = _target
