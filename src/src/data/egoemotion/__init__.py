"""Compatibility shim: ``src.data.egoemotion`` is now ``mac.data.egoemotion``.

Kept so pre-merge imports keep working. Scheduled for removal;
see docs/MERGE-MAP.md.
"""

from mac.data.egoemotion import *  # noqa: F401,F403
