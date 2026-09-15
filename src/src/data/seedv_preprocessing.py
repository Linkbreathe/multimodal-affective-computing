"""Compatibility shim: ``src.data.seedv_preprocessing`` is now ``mac.preprocessing.seedv``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.preprocessing.seedv as _target

sys.modules[__name__] = _target
