"""Compatibility shim: ``real_time_ml.policy.recommender`` is now ``mac.realtime.policy.recommender``.

Aliases the real module so both names yield the same objects.
Scheduled for removal; see docs/MERGE-MAP.md.
"""

import sys

import mac.realtime.policy.recommender as _target

sys.modules[__name__] = _target
