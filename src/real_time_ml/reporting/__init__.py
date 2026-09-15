"""Compatibility shim: ``real_time_ml.reporting`` is now ``mac.reporting``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.reporting import *  # noqa: F401,F403
