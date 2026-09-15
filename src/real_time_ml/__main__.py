"""Compatibility shim: ``real_time_ml.__main__`` is now ``mac.__main__``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.__main__ as _target

sys.modules[__name__] = _target
