"""Compatibility shim: ``src.encoders.extract`` is now ``mac.encoders.extract``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.extract as _target

sys.modules[__name__] = _target
