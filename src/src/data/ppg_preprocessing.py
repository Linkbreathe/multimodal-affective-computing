"""Compatibility shim: ``src.data.ppg_preprocessing`` is now ``mac.preprocessing.ppg``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.preprocessing.ppg as _target

sys.modules[__name__] = _target
