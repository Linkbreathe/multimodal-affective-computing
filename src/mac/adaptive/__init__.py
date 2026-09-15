"""Adaptive experiments, kept in two deliberately separate namespaces.

``offline`` replays recorded sessions through frozen models (non-interventional).
``control`` is the live control runtime that can issue commands to Unity.
They are not interchangeable; see the repository README before using either.
"""
