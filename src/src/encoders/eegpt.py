"""Compatibility shim: ``src.encoders.eegpt`` is now ``mac.encoders.eegpt``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.eegpt as _target

sys.modules[__name__] = _target
