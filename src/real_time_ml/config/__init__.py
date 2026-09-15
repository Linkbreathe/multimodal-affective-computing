"""Compatibility shim: ``real_time_ml.config`` is now ``mac.config``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.config import *  # noqa: F401,F403
