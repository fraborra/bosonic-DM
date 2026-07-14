# ruff: noqa: F403
# Copyright (C) 2025 Francesco Francesco
#
"""Shim module for backward compatibility. Imports all symbols from the new `bosonic_dm` package."""

from __future__ import annotations

from bosonic_dm._legacy import *
from bosonic_dm.efficiency import *
from bosonic_dm.io import *
from bosonic_dm.plotting.aoe import *
from bosonic_dm.stats import *
from bosonic_dm.utils import *
