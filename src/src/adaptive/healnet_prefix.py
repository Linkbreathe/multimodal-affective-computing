"""Compatibility shim: ``src.adaptive.healnet_prefix`` is now ``mac.adaptive.offline.healnet_prefix``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.adaptive.offline.healnet_prefix as _target

sys.modules[__name__] = _target
