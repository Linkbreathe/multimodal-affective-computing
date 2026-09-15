"""Compatibility shim: ``src.encoders.pulse_ppg`` is now ``mac.encoders.pulse_ppg``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.pulse_ppg as _target

sys.modules[__name__] = _target
