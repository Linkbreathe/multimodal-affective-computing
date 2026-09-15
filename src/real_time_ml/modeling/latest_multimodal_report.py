"""Compatibility shim: ``real_time_ml.modeling.latest_multimodal_report`` is now ``mac.reporting.latest_multimodal_report``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.reporting.latest_multimodal_report as _target

sys.modules[__name__] = _target
