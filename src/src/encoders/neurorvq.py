"""Compatibility shim: ``src.encoders.neurorvq`` is now ``mac.encoders.neurorvq``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.encoders.neurorvq as _target

sys.modules[__name__] = _target
