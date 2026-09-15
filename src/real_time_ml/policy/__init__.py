"""Compatibility shim: ``real_time_ml.policy`` is now ``mac.realtime.policy``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.realtime.policy import *  # noqa: F401,F403
