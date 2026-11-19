# Copyright (C) 2026 Francesco Borra
#

"""Single-Gaussian DM signal templates on top of the fixed background.

For DM mass ``m`` and detector group ``g`` the expected counts in bin ``i`` are

    lambda_gi(m) = B_hat_gi + R_m * S_gi(m),
    S_gi(m)      = X_g * eps_g(m) * P_gi(m),

where ``B_hat`` is the fixed background prediction from the broad Stan fit,
``R_m`` is the signal rate in counts/(kg yr), ``X_g`` is the group exposure in
kg yr, ``eps_g`` is the exposure-weighted group signal efficiency, and
``P_gi`` is the probability of a Gaussian centred on ``m + delta_g(m)`` in
bin ``i``, integrated exactly over the bin edges. The Gaussian width is
``FWHM_g(m) / (2 sqrt(2 ln 2))``. ``FWHM_g`` is moment matched: its square is
the exposure-weighted mean of the squared detector/run resolutions
``a + b m``, so the single Gaussian has the variance of the summed
detector/run Gaussians (``fitting-decisions.md``, with equal efficiencies and
equal centroids within a group). The optional
calibration offset ``delta_g(m)`` is the exposure-weighted mean of the
detector/run calibration residuals (calibrated minus literature peak energy)
interpolated at ``m``; it is a fixed input, zero when no residuals are given.

The efficiency is defined inside a ``+-k FWHM`` window around the peak, with
``k = 2`` by default (``+-4.7 sigma``), so Gaussian containment in that window
is essentially one. It is not corrected again here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import ndtr

from bosonic_dm.constants import FWHM_TO_SIGMA
from bosonic_dm.cuts import compute_group_exposure, matches_period_run_selection
from bosonic_dm.efficiency import aggregate_run_aware_efficiencies
from bosonic_dm.fitting.stan_utils import sha256_file
from bosonic_dm.resolution import compute_fwhm, propagate_resolution_uncertainty
from bosonic_dm.stats import compute_weighted_uncertainty, weighted_mean
from bosonic_dm.yaml_io import read_yaml

logger = logging.getLogger(__name__)

_EDGE_ATOL_KEV = 1e-9

# Entry key with the calibration-peak literature energies, in keV.
RESIDUAL_PEAKS_KEY = "cal_peaks_keV"
# Entry key with calibrated minus literature peak energies, in keV.
RESIDUALS_KEY = "cal_residuals_keV"
# Optional entry key with the residual uncertainties, in keV.
RESIDUALS_UNC_KEY = "cal_residuals_unc_keV"


@dataclass(frozen=True)
class GroupEfficiency:
    """Signal efficiency of one detector group on the simulated mass grid."""

    energies_keV: np.ndarray
    efficiency: np.ndarray
    efficiency_unc: np.ndarray
    exposure_kg_yr: np.ndarray

    def at(self, mass_keV: float) -> tuple[float, float, float]:
        """Return interpolated efficiency, its uncertainty, and the exposure."""
        return (
            interpolate_efficiency(self.energies_keV, self.efficiency, mass_keV),
            interpolate_efficiency(self.energies_keV, self.efficiency_unc, mass_keV),
            interpolate_efficiency(self.energies_keV, self.exposure_kg_yr, mass_keV),
        )


@dataclass(frozen=True)
class BackgroundPrediction:
    """Fixed background prediction ``B_hat`` of one detector group."""

    group: str
    bin_edges: np.ndarray
    counts: np.ndarray
    b_hat: np.ndarray
    used_in_fit: np.ndarray
    passed_diagnostics: bool
    files_sha256: dict[str, str]


@dataclass(frozen=True)
class GroupSignalInputs:
    """Window counts, fixed background, and signal template of one group."""

    counts: np.ndarray
    b_hat: np.ndarray
    signal_template: np.ndarray
    use_bin: np.ndarray
    exposure_kg_yr: float
    efficiency: float
    efficiency_unc: float
    efficiency_exposure_kg_yr: float
    fwhm_keV: float
    fwhm_unc_keV: float
    sigma_keV: float
    centroid_keV: float
    centroid_offset_keV: float
    centroid_offset_unc_keV: float
    centroid_offset_extrapolated: bool
    window_containment: float


@dataclass(frozen=True)
class SignalWindow:
    """Inputs of the fixed-background signal fit at one DM mass."""

    mass_keV: float
    half_width_keV: float
    bin_edges: np.ndarray
    groups: dict[str, GroupSignalInputs]
    background_sha256: dict[str, dict[str, str]]

    @property
    def group_names(self) -> list[str]:
        """Group names, in the row order of the stacked arrays."""
        return list(self.groups)

    def stacked(self, field: str) -> np.ndarray:
        """Stack a per-bin field over groups, shape ``(groups, bins)``."""
        return np.vstack([getattr(inputs, field) for inputs in self.groups.values()])


def _group_entries(
    nested: Mapping,
    group_dict: Mapping[str, str | Mapping[str, str | Sequence[str]]],
) -> Iterator[tuple[str, str, str, Mapping]]:
    """Yield usable ``(period, run, detector, entry)`` selected by a group rule."""
    for period, runs in nested.items():
        for run, detectors in runs.items():
            for detector, info in detectors.items():
                if detector not in group_dict or info.get("usability") != "on":
                    continue
                if matches_period_run_selection(
                    str(period), str(run), group_dict[detector]
                ):
                    yield str(period), str(run), str(detector), info


def group_fwhm(
    eres_dict: Mapping,
    group_dict: Mapping[str, str | Mapping[str, str | Sequence[str]]],
    energy_keV: float,
    exposure_key: str = "expo",
) -> tuple[float, float]:
    """Return the moment-matched FWHM of a detector group and its uncertainty.

    Every usable ``period/run/detector`` entry of ``eres_dict`` selected by
    ``group_dict`` (same rules as ``compute_group_exposure``) contributes a
    Gaussian of width ``FWHM_j = sqrt(a + b E)`` with weight equal to its
    exposure ``X_j``. The returned width matches the variance of their sum:
    ``FWHM_g**2 = sum(X_j FWHM_j**2) / sum(X_j)``. It assumes equal
    efficiencies and equal centroids within the group, and it is never smaller
    than the exposure-weighted mean FWHM.

    The uncertainty is the one of the exposure-weighted mean FWHM: the
    propagated calibration uncertainties (``ab_corr`` is the a-b covariance)
    combined with the weighted scatter, as in ``compute_weighted_uncertainty``.

    Parameters
    ----------
    eres_dict
        Nested ``{period: {run: {detector: {usability, expo, a, b, a_unc,
        b_unc, ab_corr}}}}`` dictionary.
    group_dict
        Detector selection of one group, as in ``groups_dict.yaml``.
    energy_keV
        Energy at which the resolution is evaluated.
    exposure_key
        Entry key holding the exposure used as weight.

    Returns
    -------
    tuple[float, float]
        Moment-matched FWHM and its uncertainty, in keV.

    Raises
    ------
    ValueError
        If no usable entry with resolution parameters is selected.
    """
    rows: list[tuple[float, float, float, float, float, float]] = []
    for period, run, detector, info in _group_entries(eres_dict, group_dict):
        if any(key not in info for key in ("a", "b", exposure_key)):
            logger.warning(
                "No resolution parameters for %s in %s-%s; skipped.",
                detector,
                period,
                run,
            )
            continue
        rows.append(
            (
                float(info[exposure_key]),
                float(info["a"]),
                float(info["b"]),
                float(info.get("a_unc", 0.0)),
                float(info.get("b_unc", 0.0)),
                float(info.get("ab_corr", 0.0)),
            )
        )
    if not rows:
        msg = "No usable detector/run entry with resolution parameters in the group."
        raise ValueError(msg)

    expo, a, b, a_unc, b_unc, ab_cov = np.asarray(rows, dtype=float).T
    fwhm = compute_fwhm(a, b, energy_keV)
    fwhm_unc = propagate_resolution_uncertainty(a, b, a_unc, b_unc, ab_cov, energy_keV)
    matched = float(np.sqrt(weighted_mean(fwhm**2, expo)))
    if not np.isfinite(matched) or matched <= 0:
        msg = f"Group FWHM at {energy_keV} keV is not positive: {matched}."
        raise ValueError(msg)
    mean = weighted_mean(fwhm, expo)
    return matched, float(compute_weighted_uncertainty(expo, fwhm, mean, fwhm_unc))


def calibration_offset(
    peaks_keV: Sequence[float] | np.ndarray,
    residuals_keV: Sequence[float] | np.ndarray,
    energy_keV: float,
    residuals_unc_keV: Sequence[float] | np.ndarray | None = None,
) -> tuple[float, float, bool]:
    """Interpolate one detector/run calibration residual curve at an energy.

    Residuals are calibrated minus literature peak energies. They are
    interpolated linearly between calibration peaks and held at the nearest
    peak value outside the peak range, which is reported as extrapolated.

    Returns
    -------
    tuple[float, float, bool]
        Offset and its uncertainty in keV, and whether ``energy_keV`` lies
        outside the calibration-peak range.
    """
    peaks = np.asarray(peaks_keV, dtype=float)
    residuals = np.asarray(residuals_keV, dtype=float)
    uncertainties = (
        np.zeros_like(residuals)
        if residuals_unc_keV is None
        else np.asarray(residuals_unc_keV, dtype=float)
    )
    if peaks.ndim != 1 or not (peaks.shape == residuals.shape == uncertainties.shape):
        msg = "Calibration peaks, residuals, and uncertainties must have equal length."
        raise ValueError(msg)
    valid = np.isfinite(peaks) & np.isfinite(residuals) & np.isfinite(uncertainties)
    if not valid.any():
        msg = "No finite calibration residual."
        raise ValueError(msg)
    order = np.argsort(peaks[valid])
    peaks, residuals, uncertainties = (
        values[valid][order] for values in (peaks, residuals, uncertainties)
    )
    if np.any(np.diff(peaks) <= 0):
        msg = "Calibration peak energies must be distinct."
        raise ValueError(msg)
    return (
        float(np.interp(energy_keV, peaks, residuals)),
        float(np.interp(energy_keV, peaks, uncertainties)),
        bool(energy_keV < peaks[0] or energy_keV > peaks[-1]),
    )


def group_centroid_offset(
    residuals_dict: Mapping,
    group_dict: Mapping[str, str | Mapping[str, str | Sequence[str]]],
    energy_keV: float,
    exposure_key: str = "expo",
) -> tuple[float, float, bool]:
    """Return the exposure-weighted calibration offset of a detector group.

    Every usable ``period/run/detector`` entry selected by ``group_dict`` (same
    rules as ``group_fwhm``) contributes its residual curve interpolated at
    ``energy_keV`` (``calibration_offset``), weighted by its exposure. Entries
    without residuals are skipped with a warning; the weights are then
    renormalized over the entries with residuals.

    Parameters
    ----------
    residuals_dict
        Nested ``{period: {run: {detector: {usability, expo, cal_peaks_keV,
        cal_residuals_keV[, cal_residuals_unc_keV]}}}}`` dictionary. It can be
        the exposure/resolution dictionary extended with these keys.
    group_dict
        Detector selection of one group, as in ``groups_dict.yaml``.
    energy_keV
        Energy at which the offset is evaluated.
    exposure_key
        Entry key holding the exposure used as weight.

    Returns
    -------
    tuple[float, float, bool]
        Offset and its uncertainty in keV (propagated residual uncertainties
        combined with the weighted scatter, as in ``group_fwhm``), and whether
        any contributing curve was extrapolated.

    Raises
    ------
    ValueError
        If no selected entry has calibration residuals.
    """
    rows: list[tuple[float, float, float]] = []
    extrapolated = False
    missing_exposure = 0.0
    for period, run, detector, info in _group_entries(residuals_dict, group_dict):
        if RESIDUAL_PEAKS_KEY not in info or RESIDUALS_KEY not in info:
            missing_exposure += float(info.get(exposure_key, 0.0))
            logger.warning(
                "No calibration residuals for %s in %s-%s; skipped.",
                detector,
                period,
                run,
            )
            continue
        offset, offset_unc, outside = calibration_offset(
            info[RESIDUAL_PEAKS_KEY],
            info[RESIDUALS_KEY],
            energy_keV,
            info.get(RESIDUALS_UNC_KEY),
        )
        extrapolated |= outside
        rows.append((float(info[exposure_key]), offset, offset_unc))
    if not rows:
        msg = "No usable detector/run entry with calibration residuals in the group."
        raise ValueError(msg)

    expo, offsets, uncertainties = np.asarray(rows, dtype=float).T
    if missing_exposure > 0:
        logger.warning(
            "Calibration residuals missing for %.2f%% of the group exposure.",
            100 * missing_exposure / (missing_exposure + expo.sum()),
        )
    mean = weighted_mean(offsets, expo)
    unc = compute_weighted_uncertainty(expo, offsets, mean, uncertainties)
    return float(mean), float(unc), extrapolated


def group_efficiency(
    efficiency: pl.DataFrame | Mapping,
    groups_dict: Mapping[str, Mapping[str, str | Mapping[str, str | Sequence[str]]]],
    selection: str = "all",
) -> dict[str, GroupEfficiency]:
    """Aggregate run-level signal efficiencies into detector groups.

    Parameters
    ----------
    efficiency
        Efficiency table written by the efficiencies stage (read it with
        ``bosonic_dm.efficiency.load_efficiency_table``), or the nested
        output of ``compute_efficiency_from_lazyframe``. Run-level results
        are required, since group membership can depend on the period.
    groups_dict
        Detector-group definitions, as in ``groups_dict.yaml``.
    selection
        Efficiency selection. ``"all"`` (LAr veto included when configured)
        matches ``passes_analysis`` on the background data.

    Returns
    -------
    dict[str, GroupEfficiency]
        Exposure-weighted efficiency per group on the simulated mass grid.

    Raises
    ------
    ValueError
        If a nested input has no run-level results or a group has no valid
        efficiency.
    """
    has_run_results = isinstance(efficiency, pl.DataFrame) or any(
        isinstance(result, Mapping) and "period_runs" in result
        for detectors in efficiency.values()
        for result in detectors.values()
    )
    if not has_run_results:
        msg = (
            "The efficiency dictionary has no run-level 'period_runs' entries; "
            "regenerate it with the current efficiency stage."
        )
        raise ValueError(msg)

    by_energy = aggregate_run_aware_efficiencies(
        efficiency,
        selection,
        group_by="detector_group",
        detector_groups=groups_dict,
    )
    output: dict[str, GroupEfficiency] = {}
    for group in groups_dict:
        rows = sorted(
            (float(energy), entry["value"], entry["unc"], entry["exposure"])
            for energy, groups in by_energy.items()
            if (entry := groups.get(group)) is not None
        )
        if not rows:
            msg = f"No valid {selection!r} efficiency for group {group!r}."
            raise ValueError(msg)
        energies, values, uncertainties, exposures = np.asarray(rows, dtype=float).T
        output[group] = GroupEfficiency(energies, values, uncertainties, exposures)
    return output


def interpolate_efficiency(
    energies_keV: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    mass_keV: float,
) -> float:
    """Linearly interpolate a quantity on the simulated mass grid.

    Raises
    ------
    ValueError
        If ``mass_keV`` lies outside the simulated range; no extrapolation is
        performed.
    """
    energies = np.asarray(energies_keV, dtype=float)
    values = np.asarray(values, dtype=float)
    if energies.ndim != 1 or energies.shape != values.shape or len(energies) == 0:
        msg = "energies_keV and values must be nonempty 1D arrays of equal length."
        raise ValueError(msg)
    if np.any(np.diff(energies) <= 0):
        msg = "energies_keV must increase strictly."
        raise ValueError(msg)
    if not energies[0] <= mass_keV <= energies[-1]:
        msg = (
            f"Mass {mass_keV} keV is outside the simulated range "
            f"[{energies[0]}, {energies[-1]}] keV; extrapolation is not allowed."
        )
        raise ValueError(msg)
    return float(np.interp(mass_keV, energies, values))


def gaussian_bin_probabilities(
    edges: Sequence[float] | np.ndarray, mu: float, sigma: float
) -> np.ndarray:
    """Return the probability of a normal distribution in every bin.

    Each bin probability is a difference of normal CDFs at the bin edges. The
    upper tail uses survival functions to avoid cancellation.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        msg = "edges must be a 1D array of at least two strictly increasing values."
        raise ValueError(msg)
    if not np.isfinite(sigma) or sigma <= 0:
        msg = f"sigma must be finite and positive; got {sigma}."
        raise ValueError(msg)
    z = (edges - mu) / sigma
    lower, upper = z[:-1], z[1:]
    return np.where(lower >= 0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower))


def load_background_prediction(
    directory: str | Path, group: str | None = None
) -> BackgroundPrediction:
    """Load ``B_hat`` written by ``save_background_fit``.

    Parameters
    ----------
    directory
        Fit output directory containing ``bhat.parquet`` and ``summary.yaml``.
    group
        Group name. Defaults to ``provenance.group`` in ``summary.yaml``, or
        the directory name.
    """
    directory = Path(directory)
    bhat_path = directory / "bhat.parquet"
    summary_path = directory / "summary.yaml"
    table = pl.read_parquet(bhat_path)
    summary = read_yaml(summary_path)
    if not isinstance(summary, Mapping):
        msg = f"{summary_path} does not contain a mapping."
        raise TypeError(msg)

    lo = table["bin_lo_keV"].to_numpy()
    hi = table["bin_hi_keV"].to_numpy()
    if not np.allclose(lo[1:], hi[:-1], rtol=0, atol=_EDGE_ATOL_KEV):
        msg = f"Bins in {bhat_path} are not contiguous."
        raise ValueError(msg)
    if group is None:
        group = str((summary.get("provenance") or {}).get("group", directory.name))
    passed = bool(summary.get("passed_diagnostics", False))
    if not passed:
        logger.warning("Background fit for %s did not pass its diagnostics.", group)
    return BackgroundPrediction(
        group=group,
        bin_edges=np.append(lo, hi[-1]),
        counts=table["counts"].to_numpy().astype(np.int64),
        b_hat=table["B_hat"].to_numpy().astype(float),
        used_in_fit=table["used_in_fit"].to_numpy().astype(bool),
        passed_diagnostics=passed,
        files_sha256={
            bhat_path.name: sha256_file(bhat_path),
            summary_path.name: sha256_file(summary_path),
        },
    )


def build_signal_window(
    mass_keV: float,
    backgrounds: Mapping[str, BackgroundPrediction],
    *,
    eres_dict: Mapping,
    groups_dict: Mapping[str, Mapping[str, str | Mapping[str, str | Sequence[str]]]],
    group_efficiencies: Mapping[str, GroupEfficiency],
    half_width_keV: float = 12.5,
    exposure_rtol: float = 0.01,
    calibration_residuals: Mapping | None = None,
    exposure_key: str = "expo",
) -> SignalWindow:
    """Cut the fit window around one mass and build every group's signal template.

    The window contains the bins lying entirely inside
    ``[mass - half_width, mass + half_width]``; with 1 keV bins on integer
    edges and the default half-width this is 24 bins. Bins left out of the
    background likelihood (masked gamma-peak windows) are also left out of
    the signal likelihood, because ``B_hat`` there excludes the peak.

    Parameters
    ----------
    mass_keV
        DM mass hypothesis; also the Gaussian centroid.
    backgrounds
        Fixed background prediction per group. All groups must share bin edges.
    eres_dict
        Exposure and resolution dictionary used for the background spectra.
    groups_dict
        Detector-group definitions.
    group_efficiencies
        Output of ``group_efficiency``.
    half_width_keV
        Half-width of the fit window.
    exposure_rtol
        Relative tolerance between the group exposure and the exposure behind
        the efficiency; larger differences are logged as warnings.
    calibration_residuals
        Optional per-detector/run calibration residuals (see
        ``group_centroid_offset``). When given, every group's Gaussian is
        centred on ``mass + delta_g(mass)``; otherwise on ``mass``. The fit
        window stays centred on ``mass``.
    exposure_key
        Entry key holding the exposure in ``eres_dict`` and
        ``calibration_residuals``. Use the key the background spectra were
        prepared with.

    Returns
    -------
    SignalWindow
        Per-group counts, ``B_hat``, signal template, and response inputs.

    Raises
    ------
    ValueError
        If the bin edges differ between groups, the window is outside the
        background range, or no bin of any group enters the likelihood.
    """
    if not backgrounds:
        msg = "At least one background prediction is required."
        raise ValueError(msg)
    if not np.isfinite(half_width_keV) or half_width_keV <= 0:
        msg = "half_width_keV must be finite and positive."
        raise ValueError(msg)

    reference = next(iter(backgrounds.values())).bin_edges
    for name, background in backgrounds.items():
        if background.bin_edges.shape != reference.shape or not np.allclose(
            background.bin_edges, reference, rtol=0, atol=_EDGE_ATOL_KEV
        ):
            msg = f"Bin edges of group {name!r} differ from the other groups."
            raise ValueError(msg)

    low, high = mass_keV - half_width_keV, mass_keV + half_width_keV
    if low < reference[0] - _EDGE_ATOL_KEV or high > reference[-1] + _EDGE_ATOL_KEV:
        msg = (
            f"Window [{low}, {high}] keV is outside the background range "
            f"[{reference[0]}, {reference[-1]}] keV."
        )
        raise ValueError(msg)
    selected = (reference[:-1] >= low - _EDGE_ATOL_KEV) & (
        reference[1:] <= high + _EDGE_ATOL_KEV
    )
    if not selected.any():
        msg = f"No bin lies entirely inside [{low}, {high}] keV."
        raise ValueError(msg)
    indices = np.flatnonzero(selected)
    edges = reference[indices[0] : indices[-1] + 2]

    groups: dict[str, GroupSignalInputs] = {}
    for name, background in backgrounds.items():
        exposure = compute_group_exposure(eres_dict, groups_dict[name], exposure_key)
        if not np.isfinite(exposure) or exposure <= 0:
            msg = f"Group {name!r} has no positive exposure."
            raise ValueError(msg)
        efficiency, efficiency_unc, efficiency_exposure = group_efficiencies[name].at(
            mass_keV
        )
        if abs(efficiency_exposure - exposure) > exposure_rtol * exposure:
            logger.warning(
                "Group %s: exposure behind the efficiency (%.4g kg yr) differs from "
                "the data exposure (%.4g kg yr).",
                name,
                efficiency_exposure,
                exposure,
            )
        fwhm, fwhm_unc = group_fwhm(
            eres_dict, groups_dict[name], mass_keV, exposure_key
        )
        sigma = fwhm * FWHM_TO_SIGMA
        if calibration_residuals is None:
            offset, offset_unc, extrapolated = 0.0, 0.0, False
        else:
            offset, offset_unc, extrapolated = group_centroid_offset(
                calibration_residuals, groups_dict[name], mass_keV, exposure_key
            )
            if extrapolated:
                logger.warning(
                    "Group %s: %s keV lies outside some calibration-peak ranges; "
                    "the nearest residual is used.",
                    name,
                    mass_keV,
                )
        centroid = mass_keV + offset
        probabilities = gaussian_bin_probabilities(edges, centroid, sigma)
        groups[name] = GroupSignalInputs(
            counts=background.counts[indices].copy(),
            b_hat=background.b_hat[indices].copy(),
            signal_template=exposure * efficiency * probabilities,
            use_bin=background.used_in_fit[indices].copy(),
            exposure_kg_yr=float(exposure),
            efficiency=efficiency,
            efficiency_unc=efficiency_unc,
            efficiency_exposure_kg_yr=efficiency_exposure,
            fwhm_keV=fwhm,
            fwhm_unc_keV=fwhm_unc,
            sigma_keV=float(sigma),
            centroid_keV=float(centroid),
            centroid_offset_keV=offset,
            centroid_offset_unc_keV=offset_unc,
            centroid_offset_extrapolated=extrapolated,
            window_containment=float(probabilities.sum()),
        )

    if not any(inputs.use_bin.any() for inputs in groups.values()):
        msg = f"Every bin of the {mass_keV} keV window is masked in every group."
        raise ValueError(msg)
    return SignalWindow(
        mass_keV=float(mass_keV),
        half_width_keV=float(half_width_keV),
        bin_edges=edges,
        groups=groups,
        background_sha256={
            name: background.files_sha256 for name, background in backgrounds.items()
        },
    )
