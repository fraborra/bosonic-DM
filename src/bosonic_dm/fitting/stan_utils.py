# Copyright (C) 2026 Francesco Borra
#

"""Stan file locations and provenance records shared by the fits."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cmdstanpy

# The Stan models live in ``stan/`` at the repository root, outside the
# package, so they are found only with an editable installation.
STAN_DIR = Path(__file__).resolve().parents[3] / "stan"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def software_record(stan_file: str | Path) -> dict[str, str]:
    """Return the Stan file name and hash with the CmdStan and CmdStanPy versions."""
    import cmdstanpy  # noqa: PLC0415

    return {
        "stan_file": Path(stan_file).name,
        "stan_file_sha256": sha256_file(stan_file),
        "cmdstan_version": ".".join(map(str, cmdstanpy.cmdstan_version() or ())),
        "cmdstanpy_version": cmdstanpy.__version__,
    }


def sampler_record(fit: cmdstanpy.CmdStanMCMC) -> dict[str, Any]:
    """Return the seed, chain count, and sampler settings CmdStan recorded for a fit."""
    config = fit.metadata.cmdstan_config
    return {
        "seed": config.get("seed"),
        "chains": int(fit.chains),
        "num_warmup": config.get("num_warmup"),
        "num_samples": config.get("num_samples"),
        "adapt_delta": config.get("delta"),
    }
