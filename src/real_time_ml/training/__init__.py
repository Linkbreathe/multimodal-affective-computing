"""Compatibility shim: ``real_time_ml.training`` is now ``mac.training``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.training import *  # noqa: F401,F403
