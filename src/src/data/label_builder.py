"""Compatibility shim: ``src.data.label_builder`` is now ``mac.data.label_builder``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.data.label_builder as _target

sys.modules[__name__] = _target
