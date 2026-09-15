"""Compatibility shim: ``src.utils.logging_setup`` is now ``mac.utils.logging_setup``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.utils.logging_setup as _target

sys.modules[__name__] = _target
