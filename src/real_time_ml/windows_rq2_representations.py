"""Compatibility shim: ``real_time_ml.windows_rq2_representations`` is now ``mac.windows_rq2_representations``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.windows_rq2_representations as _target

sys.modules[__name__] = _target
