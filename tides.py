#!/usr/bin/env python3
"""
Tide Prediction API
===================

Predict astronomical tide heights at any (latitude, longitude, datetime).

Quick Start
-----------
>>> from tides import predict
>>> from datetime import datetime

>>> # Predict tide at San Francisco
>>> tide_meters = predict(37.8067, -122.465, datetime(2026, 1, 3, 17, 34))
>>> print(f"Tide height: {tide_meters:.2f} m")
Tide height: 1.23 m

Installation
------------
pip install "pyTMD[all]" timescale pandas matplotlib scipy

Author: Ecological Intelligence Inc. (Ecodash.ai)
License: MIT
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, List, Optional

import pyTMD
import pyTMD.io
import pyTMD.compute
import pyTMD.datasets
import platformdirs

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default cache directory for tide models
MODEL_DIR = platformdirs.user_cache_dir('pytmd')

# Default model (can be changed globally)
DEFAULT_MODEL = 'GOT4.10_nc'

# EOT20 download URL (DOI: 10.17882/79489)
EOT20_URL = "https://www.seanoe.org/data/00683/79489/data/85762.zip"

# Track if resolution warning has been shown
_RESOLUTION_WARNING_SHOWN = False


# =============================================================================
# MAIN API
# =============================================================================

def predict(
    lat: float,
    lon: float,
    time: datetime,
    model: str = None
) -> float:
    """
    Predict tide height at a specific location and time.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees (-90 to 90)
    lon : float
        Longitude in decimal degrees (-180 to 180)
    time : datetime
        UTC datetime for prediction
    model : str, optional
        Tide model to use:
        - 'GOT4.10_nc' (default): 0.5° resolution, auto-downloads
        - 'EOT20': 0.125° resolution, requires download_eot20()

    Returns
    -------
    float
        Tide elevation in meters relative to model datum

    Examples
    --------
    >>> from tides import predict
    >>> from datetime import datetime

    >>> # San Francisco king tide
    >>> predict(37.8067, -122.465, datetime(2026, 1, 3, 17, 34))
    1.2329

    >>> # Tokyo Bay
    >>> predict(35.6762, 139.6503, datetime(2026, 1, 3, 12, 0))
    0.8451
    """
    if model is None:
        model = DEFAULT_MODEL

    _check_resolution(model)
    _ensure_model(model)

    x = np.array([lon])
    y = np.array([lat])
    delta_time = np.array([np.datetime64(time)])

    tide = pyTMD.compute.tide_elevations(
        x, y, delta_time,
        model=model,
        directory=MODEL_DIR,
        type='drift',
        standard='datetime',
        method='linear',
        extrapolate=True,
        cutoff=np.inf
    )

    return float(tide.values[0])


def predict_timeseries(
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    interval_minutes: int = 10,
    model: str = None
) -> pd.DataFrame:
    """
    Predict tide heights over a time period.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
    start : datetime
        Start of prediction period (UTC)
    end : datetime
        End of prediction period (UTC)
    interval_minutes : int
        Time step in minutes (default: 10)
    model : str, optional
        Tide model to use

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime_utc', 'tide_m']

    Examples
    --------
    >>> from tides import predict_timeseries
    >>> from datetime import datetime

    >>> df = predict_timeseries(
    ...     37.8067, -122.465,
    ...     datetime(2026, 1, 1),
    ...     datetime(2026, 1, 7)
    ... )
    >>> df.head()
    """
    if model is None:
        model = DEFAULT_MODEL

    _check_resolution(model)
    _ensure_model(model)

    n_steps = int((end - start).total_seconds() / (interval_minutes * 60)) + 1
    times = [start + timedelta(minutes=i * interval_minutes) for i in range(n_steps)]
    delta_time = np.array([np.datetime64(t) for t in times])

    x = np.full(n_steps, lon)
    y = np.full(n_steps, lat)

    tides = pyTMD.compute.tide_elevations(
        x, y, delta_time,
        model=model,
        directory=MODEL_DIR,
        type='drift',
        standard='datetime',
        method='linear',
        extrapolate=True,
        cutoff=np.inf
    )

    return pd.DataFrame({
        'datetime_utc': times,
        'tide_m': tides.values
    })


def predict_batch(
    lats: np.ndarray,
    lons: np.ndarray,
    time: datetime,
    model: str = None
) -> np.ndarray:
    """
    Predict tides at multiple locations for a single time.

    Parameters
    ----------
    lats : np.ndarray
        Array of latitudes
    lons : np.ndarray
        Array of longitudes
    time : datetime
        UTC datetime for all predictions
    model : str, optional
        Tide model to use

    Returns
    -------
    np.ndarray
        Array of tide elevations in meters

    Examples
    --------
    >>> from tides import predict_batch
    >>> import numpy as np

    >>> # Create a grid over San Francisco Bay
    >>> lats = np.linspace(37.4, 38.2, 10)
    >>> lons = np.linspace(-122.8, -122.0, 10)
    >>> lat_grid, lon_grid = np.meshgrid(lats, lons)
    >>> tides = predict_batch(lat_grid.ravel(), lon_grid.ravel(), datetime(2026, 1, 3, 17, 34))
    """
    if model is None:
        model = DEFAULT_MODEL

    _check_resolution(model)
    _ensure_model(model)

    n_points = len(lats)
    delta_time = np.array([np.datetime64(time)] * n_points)

    tides = pyTMD.compute.tide_elevations(
        lons, lats, delta_time,
        model=model,
        directory=MODEL_DIR,
        type='drift',
        standard='datetime',
        method='linear',
        extrapolate=True,
        cutoff=np.inf
    )

    return tides.values


# =============================================================================
# HIGH/LOW TIDE DETECTION
# =============================================================================

def find_high_low_tides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify high and low tide times from a tide time series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'datetime_utc' and 'tide_m' columns
        (as returned by predict_timeseries)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime_utc', 'tide_m', 'type']
        where type is 'HIGH' or 'LOW'

    Examples
    --------
    >>> df = predict_timeseries(37.8067, -122.465, start, end)
    >>> events = find_high_low_tides(df)
    >>> highs = events[events['type'] == 'HIGH']
    """
    from scipy.signal import argrelextrema

    tide_values = df['tide_m'].values

    high_idx = argrelextrema(tide_values, np.greater, order=10)[0]
    low_idx = argrelextrema(tide_values, np.less, order=10)[0]

    events = []

    for idx in high_idx:
        events.append({
            'datetime_utc': df.iloc[idx]['datetime_utc'],
            'tide_m': df.iloc[idx]['tide_m'],
            'type': 'HIGH'
        })

    for idx in low_idx:
        events.append({
            'datetime_utc': df.iloc[idx]['datetime_utc'],
            'tide_m': df.iloc[idx]['tide_m'],
            'type': 'LOW'
        })

    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values('datetime_utc').reset_index(drop=True)

    return events_df


# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

def download_eot20():
    """
    Download the EOT20 high-resolution tide model (~2 GB).

    EOT20 provides 0.125° (~14 km) resolution, which is 4x higher
    than the default GOT4.10 model (0.5°).

    Reference: Hart-Davis et al. (2021), https://doi.org/10.17882/79489
    """
    import urllib.request
    import zipfile
    import shutil

    eot20_dir = os.path.join(MODEL_DIR, 'EOT20')
    ocean_tides_dir = os.path.join(eot20_dir, 'ocean_tides')

    # Check if already downloaded
    if os.path.exists(ocean_tides_dir):
        nc_files = [f for f in os.listdir(ocean_tides_dir) if f.endswith('.nc')]
        if len(nc_files) >= 17:
            print(f"EOT20 already downloaded at: {eot20_dir}")
            return eot20_dir

    print("=" * 60)
    print("Downloading EOT20 Ocean Tide Model")
    print("=" * 60)
    print(f"Source: SEANOE (DOI: 10.17882/79489)")
    print(f"Resolution: 1/8° (~14 km)")
    print(f"File size: ~2 GB")
    print("-" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    zip_path = os.path.join(MODEL_DIR, 'eot20.zip')

    print(f"Downloading from: {EOT20_URL}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb:.1f} MB)", end='', flush=True)

    urllib.request.urlretrieve(EOT20_URL, zip_path, reporthook=report_progress)
    print()

    # Extract
    print(f"Extracting to: {MODEL_DIR}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    # Handle nested zip structure
    os.makedirs(ocean_tides_dir, exist_ok=True)
    nested_zip = os.path.join(MODEL_DIR, 'ocean_tides.zip')
    if os.path.exists(nested_zip):
        print("Extracting ocean_tides.zip...")
        with zipfile.ZipFile(nested_zip, 'r') as zip_ref:
            zip_ref.extractall(ocean_tides_dir)

        # Move files from nested folder if present
        nested_folder = os.path.join(ocean_tides_dir, 'ocean_tides')
        if os.path.exists(nested_folder):
            for f in os.listdir(nested_folder):
                if f.endswith('.nc'):
                    shutil.move(os.path.join(nested_folder, f),
                               os.path.join(ocean_tides_dir, f))
            shutil.rmtree(nested_folder)

        # Clean up
        macosx = os.path.join(ocean_tides_dir, '__MACOSX')
        if os.path.exists(macosx):
            shutil.rmtree(macosx)

    os.remove(zip_path)

    nc_files = [f for f in os.listdir(ocean_tides_dir) if f.endswith('.nc')]
    print(f"EOT20 ready with {len(nc_files)} constituent files")
    print("=" * 60)

    return eot20_dir


def set_model(model_name: str):
    """
    Set the default tide model for all predictions.

    Parameters
    ----------
    model_name : str
        'GOT4.10_nc' (default) or 'EOT20'
    """
    global DEFAULT_MODEL
    DEFAULT_MODEL = model_name


def get_model_info() -> dict:
    """
    Get information about available tide models.

    Returns
    -------
    dict
        Model names, resolutions, and availability
    """
    eot20_path = os.path.join(MODEL_DIR, 'EOT20', 'ocean_tides')
    eot20_available = os.path.exists(eot20_path)

    return {
        'GOT4.10_nc': {
            'resolution': '0.5° (~55 km)',
            'source': 'NASA GSFC',
            'available': True,
            'auto_download': True
        },
        'EOT20': {
            'resolution': '0.125° (~14 km)',
            'source': 'SEANOE (Hart-Davis et al. 2021)',
            'available': eot20_available,
            'auto_download': False,
            'download_command': 'tides.download_eot20()'
        }
    }


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _check_resolution(model_name: str):
    """Issue a one-time warning if using low-resolution model."""
    global _RESOLUTION_WARNING_SHOWN
    if not _RESOLUTION_WARNING_SHOWN and model_name in ('GOT4.10_nc', 'GOT4.10'):
        import sys
        print("\n" + "=" * 60, file=sys.stderr)
        print("NOTE: Using GOT4.10 model (0.5° resolution)", file=sys.stderr)
        print("For higher coastal accuracy, use EOT20 (0.125°):", file=sys.stderr)
        print("  >>> import tides", file=sys.stderr)
        print("  >>> tides.download_eot20()", file=sys.stderr)
        print("  >>> tides.set_model('EOT20')", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)
        _RESOLUTION_WARNING_SHOWN = True


def _ensure_model(model_name: str):
    """Ensure the requested model is available."""
    if model_name == 'EOT20':
        eot20_path = os.path.join(MODEL_DIR, 'EOT20', 'ocean_tides')
        if not os.path.exists(eot20_path):
            raise FileNotFoundError(
                "EOT20 model not found. Download it first:\n"
                "  >>> import tides\n"
                "  >>> tides.download_eot20()"
            )
    elif 'GOT' in model_name:
        got_version = model_name.replace('_nc', '')
        pyTMD.datasets.fetch_gsfc_got(model=got_version, directory=MODEL_DIR)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        print("""
Command Line Usage
------------------
  python tides.py LAT LON DATETIME [--model MODEL]

Examples
--------
  python tides.py 37.8067 -122.465 "2026-01-03 17:34"
  python tides.py 37.8067 -122.465 "2026-01-03 17:34" --model EOT20

Options
-------
  --download-eot20    Download the high-resolution EOT20 model
  --model MODEL       Use specific model (GOT4.10_nc or EOT20)
""")
        sys.exit(0)

    if '--download-eot20' in sys.argv:
        download_eot20()
        sys.exit(0)

    # Parse arguments
    try:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
        dt = datetime.fromisoformat(sys.argv[3])
    except (IndexError, ValueError) as e:
        print(f"Error: {e}")
        print("Usage: python tides.py LAT LON DATETIME")
        sys.exit(1)

    model = None
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        model = sys.argv[idx + 1]

    # Predict
    tide = predict(lat, lon, dt, model=model)
    print(f"Location: ({lat}, {lon})")
    print(f"Time: {dt} UTC")
    print(f"Tide height: {tide:.3f} m ({tide * 3.28084:.2f} ft)")
