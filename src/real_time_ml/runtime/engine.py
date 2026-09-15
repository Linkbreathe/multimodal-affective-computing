"""Compatibility shim: ``real_time_ml.runtime.engine`` is now ``mac.runtime.engine``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.runtime.engine as _target

sys.modules[__name__] = _target
