"""Compatibility shim: ``src.encoders.reve`` is now ``mac.encoders.reve``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.reve as _target

sys.modules[__name__] = _target
