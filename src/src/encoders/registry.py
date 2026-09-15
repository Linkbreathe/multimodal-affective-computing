"""Compatibility shim: ``src.encoders.registry`` is now ``mac.encoders.registry``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.registry as _target

sys.modules[__name__] = _target
