"""Compatibility shim: ``src.encoders.inceptiontime`` is now ``mac.encoders.inceptiontime``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.inceptiontime as _target

sys.modules[__name__] = _target
