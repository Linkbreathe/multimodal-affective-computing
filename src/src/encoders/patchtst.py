"""Compatibility shim: ``src.encoders.patchtst`` is now ``mac.encoders.patchtst``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.patchtst as _target

sys.modules[__name__] = _target
