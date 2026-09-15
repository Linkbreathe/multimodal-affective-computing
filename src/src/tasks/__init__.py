"""Compatibility shim: ``src.tasks`` is now ``mac.tasks``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.tasks import *  # noqa: F401,F403
