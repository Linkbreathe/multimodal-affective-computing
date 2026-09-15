"""Compatibility shim: ``src.utils.reporting`` is now ``mac.reporting.experiment``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.reporting.experiment as _target

sys.modules[__name__] = _target
