"""Compatibility shim: ``src.data.ecg_preprocessing`` is now ``mac.preprocessing.ecg``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.preprocessing.ecg as _target

sys.modules[__name__] = _target
