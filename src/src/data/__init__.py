"""Compatibility shim: ``src.data`` is now ``mac.data``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.data import *  # noqa: F401,F403
