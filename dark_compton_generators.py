# ruff: noqa: F403, F405
# Copyright (C) 2025 Francesco Francesco
#
"""Shim module for backward compatibility. Imports all symbols from the new `bosonic_dm` package."""

from __future__ import annotations

from bosonic_dm.dark_compton_generators import *

if __name__ == "__main__":
    main()
