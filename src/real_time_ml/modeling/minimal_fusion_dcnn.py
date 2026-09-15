"""Compatibility shim: ``real_time_ml.modeling.minimal_fusion_dcnn`` is now ``mac.fusion.minimal_fusion_dcnn``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.fusion.minimal_fusion_dcnn as _target

sys.modules[__name__] = _target
