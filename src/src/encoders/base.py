"""Compatibility shim: ``src.encoders.base`` is now ``mac.encoders.base``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.base as _target

sys.modules[__name__] = _target
