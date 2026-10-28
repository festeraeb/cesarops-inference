"""Sentinel-2 SDB (Satellite-Derived Bathymetry) helper skeleton.

This module provides a minimal SDB computation skeleton using a Blue/Green
ratio log-model placeholder. Replace coefficients with empirically-derived
values during calibration.
"""
import numpy as np
from typing import Tuple


def compute_sdb_from_bands(blue: np.ndarray, green: np.ndarray, a0: float = 10.0, a1: float = -2.0) -> np.ndarray:
    """Compute a simple SDB depth estimate from Sentinel-2 blue/green bands.

    Model (placeholder): depth = a0 + a1 * ln(blue/green)

    Args:
        blue: numpy array of blue band reflectances (or TOA)
        green: numpy array of green band reflectances (or TOA)
        a0, a1: model coefficients (must be calibrated)

    Returns:
        depth array in meters (approximate; requires calibration)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(green > 0, blue / green, np.nan)
        ln_ratio = np.log(ratio)
        depth = a0 + a1 * ln_ratio
    return depth


def compute_deep_water_spectral_shift(red_edge: np.ndarray, coastal_aerosol: np.ndarray, a0: float = 22.0, a1: float = -3.0) -> np.ndarray:
    """Estimate deeper SDB from Red-Edge (B05) / Coastal Aerosol (B01) ratio.

    The goal is to give a deeper-water proxy where Blue/Green saturates.
    Model: depth = a0 + a1 * ln(red_edge / coastal_aerosol)

    Args:
        red_edge: band B05 array
        coastal_aerosol: band B01 array
        a0, a1: model coefficients (needs empirical calibration)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(coastal_aerosol > 0, red_edge / coastal_aerosol, np.nan)
        ln_ratio = np.log(ratio)
        depth = a0 + a1 * ln_ratio
    return depth


def compute_mussel_index(coastal_aerosol: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """Proxy 'Mussel Bloom' index: coastal aerosol vs red-edge ratio.

    Higher values may indicate plume/turbidity signatures tied to biomatter.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        idx = np.where(red_edge > 0, coastal_aerosol / red_edge, np.nan)
    return idx


def compute_huron_fog_cutter(coastal_aerosol: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """Huron deep-water fog-cutter index (B01 vs B04)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        idx = np.where(red_band > 0, (coastal_aerosol - red_band) / red_band, np.nan)
    return idx


def compute_ndci(red_edge: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Chlorophyll Index using Sentinel-2 B5/B4."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ndci = np.where((red_edge + red) > 0, (red_edge - red) / (red_edge + red), np.nan)
    return ndci


def compute_aerosol_squeeze(coastal_aerosol: np.ndarray, green: np.ndarray) -> np.ndarray:
    """Coastal aerosol to green ratio (B1/B3) for deep-bathymetry squeeze."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(green > 0, coastal_aerosol / green, np.nan)
    return ratio


def detect_sdb_deviation(depth_array: np.ndarray, baseline_depth: float, threshold_m: float = 3.0) -> bool:
    """Detect if SDB in window shows an upward jump relative to baseline."""
    if np.isnan(depth_array).all():
        return False
    med = float(np.nanmedian(depth_array))
    return (baseline_depth - med) >= threshold_m


def detect_vertical_pulse(fog_index: np.ndarray, threshold_pct: float = 0.02, pixel_size_m: float = 10.0) -> bool:
    """Detect coherent 10m 'pulse' 2% brighter than local background."""
    if fog_index.size == 0 or np.isnan(fog_index).all():
        return False
    local_med = float(np.nanmedian(fog_index))
    pulse_thresh = local_med * (1.0 + threshold_pct)
    high_pixels = np.sum(fog_index > pulse_thresh)
    return high_pixels >= 1


def detect_floor_jump(depth_array: np.ndarray, window_m: Tuple[int, int] = (60, 180), threshold_m: float = 2.0) -> bool:
    """Detect if a localized 'floor jump' exists inside a rectangle window.

    Args:
        depth_array: 2D numpy array of depths (meters)
        window_m: target rectangle short/long side in meters (approx mapping to pixels handled externally)
        threshold_m: vertical deviation threshold to flag (meters)

    Returns:
        True if a floor jump exceeds threshold within the window, else False.
    """
    # Placeholder: a real implementation should map meters -> pixels and
    # analyze local statistics (median filtering, change-detection, etc.).
    if np.isnan(depth_array).all():
        return False
    dmin = np.nanmin(depth_array)
    dmax = np.nanmax(depth_array)
    return (dmax - dmin) >= threshold_m


if __name__ == "__main__":
    import numpy as _np
    blue = _np.array([[0.05, 0.04], [0.03, 0.02]])
    green = _np.array([[0.03, 0.03], [0.02, 0.02]])
    depth = compute_sdb_from_bands(blue, green)
    print("Depth (m):", depth)
    print("Floor jump?", detect_floor_jump(depth, threshold_m=0.5))
