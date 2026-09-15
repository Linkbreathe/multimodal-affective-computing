"""Compatibility shim: ``src.utils`` is now ``mac.utils``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.utils import *  # noqa: F401,F403
