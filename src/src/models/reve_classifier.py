"""Compatibility shim: ``src.models.reve_classifier`` is now ``mac.models.reve_classifier``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.models.reve_classifier as _target

sys.modules[__name__] = _target
