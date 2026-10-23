"""Offline dataset planning and local integrity checks (Python standard library)."""

import sys

# The CLI's default audit is read-only, including Python's import cache.
sys.dont_write_bytecode = True
