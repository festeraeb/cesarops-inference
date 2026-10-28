#!/usr/bin/env python3
"""Deep Water Detection DNA pipeline.

Tasks included:
- Task A: Biological Glow (NDCI from S2 B5/B4).
- Task B: Thermal Lag (ECOSTRESS placeholder).
- Task C: Flow Disruption (SAR coherence placeholder).
- Task D: SDB Deviation (B1/B3 + baseline difference).

Usage:
    python deep_water_detection.py --lat 44.1736 --lon -81.6406
    python deep_water_detection.py --lat-start 43.8 --lat-end 43.85 --lon-start -82.5 --lon-end -82.25 --grid 5
"""
from pathlib import Path
import argparse
import math
import numpy as np
import os
import sys
import subprocess
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentinel_hunt.python.nasa_fusion import FusionScorer

import importlib.util
fetcher_path = os.path.join(str(ROOT), 'recovered', 'sentinel_fetch_and_preprocess.py')
if not os.path.exists(fetcher_path):
    raise FileNotFoundError(f'Fetcher module not found at {fetcher_path}')

spec = importlib.util.spec_from_file_location('recovered_sentinel_fetch_and_preprocess', fetcher_path)
fetcher_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher_module)
fetch_for_wreck = fetcher_module.fetch_for_wreck

from scripts.forensic.sdb import (
    compute_sdb_from_bands,
    compute_huron_fog_cutter,
    compute_mussel_index,
    compute_deep_water_spectral_shift,
    compute_ndci,
    compute_aerosol_squeeze,
    detect_vertical_pulse,
    detect_sdb_deviation,
)


def _ecostress_thermal_zscore(lat: float, lon: float):
    # Placeholder: integrate ECOSTRESS ingestion and compute temporal z-score.
    # Here, return a random to ensure this path is in the pipeline for now.
    return np.nan


def _sar_phase_coherence(lat: float, lon: float):
    # Placeholder: integrate or call sar_extract stack analysis.
    return np.nan


def evaluate_point(lat: float, lon: float, baseline_depth: float = 30.0):
    out_file = fetch_for_wreck(f"forensic_{lat:.6f}_{lon:.6f}", lat, lon)
    if out_file is None:
        return None

    arr = np.load(out_file)
    if arr.shape[0] < 6:
        return None

    blue = arr[2].astype(np.float32)
    green = arr[1].astype(np.float32)
    red = arr[0].astype(np.float32)
    b5 = arr[4].astype(np.float32)
    b1 = arr[5].astype(np.float32) if arr.shape[0] > 5 else np.nan * np.ones_like(red)

    sdb = compute_sdb_from_bands(blue, green)
    ndci = compute_ndci(b5, red)
    fog = compute_huron_fog_cutter(b1, red)
    mussel = compute_mussel_index(b1, b5)
    deep_shift = compute_deep_water_spectral_shift(b5, b1)
    aerosol_ratio = compute_aerosol_squeeze(b1, green)

    cy, cx = sdb.shape[0] // 2, sdb.shape[1] // 2
    window = sdb[max(0, cy - 40) : cy + 40, max(0, cx - 80) : cx + 80]

    sdb_med = float(np.nanmedian(window))
    ndci_med = float(np.nanmedian(ndci))
    fog_med = float(np.nanmedian(fog))
    mussel_med = float(np.nanmedian(mussel))
    deep_med = float(np.nanmedian(deep_shift))
    aerosol_med = float(np.nanmedian(aerosol_ratio))

    pulse = detect_vertical_pulse(fog)
    sdb_deviation = detect_sdb_deviation(window, baseline_depth=baseline_depth, threshold_m=5.0)

    thermal_z = _ecostress_thermal_zscore(lat, lon)
    sar_coh = _sar_phase_coherence(lat, lon)

    # Simple fused score for ranking (weights can be tuned).
    score = 0.0
    score += (sdb_med < baseline_depth) * 1.0
    if not math.isnan(ndci_med):
        score += ndci_med * 5.0
    if not math.isnan(fog_med):
        score += (1.0 if pulse else 0.0) * 4.0
    if not math.isnan(sar_coh):
        score += sar_coh * 2.0
    if sdb_deviation:
        score += 5.0

    return {
        "lat": lat,
        "lon": lon,
        "sdb_m": sdb_med,
        "sdb_ft": sdb_med * 3.28084,
        "ndci": ndci_med,
        "fog": fog_med,
        "pulse": pulse,
        "mussel": mussel_med,
        "deep_shift": deep_med,
        "aerosol_ratio": aerosol_med,
        "thermal_z": thermal_z,
        "sar_coherence": sar_coh,
        "sdb_deviation": sdb_deviation,
        "score": score,
    }


def run_grid(lat_start, lat_end, lon_start, lon_end, n_lat=5, n_lon=20):
    lats = np.linspace(lat_start, lat_end, n_lat)
    lons = np.linspace(lon_start, lon_end, n_lon)
    results = []
    for lat in lats:
        for lon in lons:
            r = evaluate_point(lat, lon)
            if r is not None:
                results.append(r)
                print(f"{lat:.6f},{lon:.6f} => score {r['score']:.2f} sdb {r['sdb_m']:.2f}m {r['sdb_ft']:.2f}ft")
    if not results:
        print("No valid results")
        return []

    results = sorted(results, key=lambda x: x['score'], reverse=True)
    print("\nTOP 10 candidates:")
    for r in results[:10]:
        print(f"{r['lat']:.6f},{r['lon']:.6f} score={r['score']:.2f} sdb={r['sdb_m']:.2f}m")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run Deep Water Detection DNA sweep.')
    parser.add_argument('--lat-start', type=float, default=43.81667, help='Latitude start')
    parser.add_argument('--lat-end', type=float, default=43.82667, help='Latitude end')
    parser.add_argument('--lon-start', type=float, default=-82.5, help='Longitude start')
    parser.add_argument('--lon-end', type=float, default=-82.25, help='Longitude end')
    parser.add_argument('--lat-steps', type=int, default=5)
    parser.add_argument('--lon-steps', type=int, default=16)
    parser.add_argument('--baseline-depth', type=float, default=36.6, help='NOAA baseline depth (m)')
    args = parser.parse_args()

    run_grid(args.lat_start, args.lat_end, args.lon_start, args.lon_end, args.lat_steps, args.lon_steps)

if __name__ == '__main__':
    main()
