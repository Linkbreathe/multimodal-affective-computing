"""Compatibility shim: ``real_time_ml.modeling.video_encoder_report`` is now ``mac.reporting.video_encoder_report``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.reporting.video_encoder_report as _target

sys.modules[__name__] = _target
