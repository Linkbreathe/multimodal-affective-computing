"""Compatibility shim: ``src.utils.tb`` is now ``mac.utils.tb``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.utils.tb as _target

sys.modules[__name__] = _target
