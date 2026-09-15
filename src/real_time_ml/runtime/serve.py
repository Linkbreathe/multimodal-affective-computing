"""Compatibility shim: ``real_time_ml.runtime.serve`` is now ``mac.runtime.serve``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.runtime.serve as _target

sys.modules[__name__] = _target
