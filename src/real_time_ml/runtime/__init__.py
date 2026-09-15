"""Compatibility shim: ``real_time_ml.runtime`` is now ``mac.runtime``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.runtime import *  # noqa: F401,F403
