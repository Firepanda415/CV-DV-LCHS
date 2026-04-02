#!/usr/bin/env python3
"""
Compatibility wrapper for the old state-prep entry point.

The active implementation lives in formed_stateprep.py.
"""

from formed_stateprep import *  # noqa: F401,F403


if __name__ == "__main__":
    from formed_stateprep import main

    main()
