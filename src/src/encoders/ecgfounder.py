"""Compatibility shim: ``src.encoders.ecgfounder`` is now ``mac.encoders.ecgfounder``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.ecgfounder as _target

sys.modules[__name__] = _target
