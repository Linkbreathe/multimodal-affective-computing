"""Compatibility shim: ``real_time_ml.realtime`` is now ``mac.realtime``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.realtime import *  # noqa: F401,F403
