"""Compatibility shim: ``real_time_ml`` is now ``mac``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac import *  # noqa: F401,F403
