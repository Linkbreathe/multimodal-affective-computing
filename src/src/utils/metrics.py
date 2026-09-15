"""Compatibility shim: ``src.utils.metrics`` is now ``mac.evaluation.metrics``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.evaluation.metrics as _target

sys.modules[__name__] = _target
