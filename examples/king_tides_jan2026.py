#!/usr/bin/env python3
"""
King Tides Case Study: San Francisco Bay, January 2026
=======================================================

The January 3-4, 2026 king tides were the highest in San Francisco Bay since 1998,
driven by the alignment of:
  - Full moon (January 4, 2026)
  - Lunar perigee (moon closest to Earth)
  - Perihelion (Earth closest to Sun on January 4)
  - Atmospheric river storm surge

This script validates pyTMD predictions against this known event.

Usage:
  python examples/king_tides_jan2026.py [--use-eot20]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import tides


def analyze_3_year_king_tides(model: str = None):
    """
    Analyze king tides over 3 years from January 4, 2023 to January 4, 2026.

    King tides occur when the Earth, Moon, and Sun align to create the
    highest astronomical tides.
    """
    print("\n" + "=" * 70)
    print("3-YEAR KING TIDE ANALYSIS: Jan 4, 2023 - Jan 4, 2026")
    print("=" * 70)

    # San Francisco location
    SF_LAT, SF_LON = 37.8067, -122.4650

    start_date = datetime(2023, 1, 4, 0, 0, 0)
    end_date = datetime(2026, 1, 4, 23, 59, 0)

    print(f"\nAnalyzing tides at San Francisco ({SF_LAT}, {SF_LON})")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("\nThis may take a minute...")

    # Generate hourly predictions for 3 years
    df = tides.predict_timeseries(
        SF_LAT, SF_LON,
        start_date, end_date,
        interval_minutes=60,
        model=model
    )

    print(f"\nGenerated {len(df)} hourly predictions")

    # Find all high tides
    events = tides.find_high_low_tides(df)
    highs = events[events['type'] == 'HIGH'].copy()

    print(f"Found {len(highs)} high tides over 3 years")

    # Add month/year columns
    highs['month'] = highs['datetime_utc'].apply(lambda x: x.month)
    highs['year'] = highs['datetime_utc'].apply(lambda x: x.year)

    # Monthly maximum tides
    print("\n" + "-" * 70)
    print("MONTHLY MAXIMUM HIGH TIDES")
    print("-" * 70)
    print(f"{'Year-Month':<12} {'Date/Time (UTC)':<22} {'Height (m)':<12} {'Height (ft)':<12}")
    print("-" * 70)

    monthly_max = []
    for year in range(2023, 2027):
        for month in range(1, 13):
            if year == 2023 and month < 1:
                continue
            if year == 2026 and month > 1:
                continue

            month_data = highs[(highs['year'] == year) & (highs['month'] == month)]
            if len(month_data) > 0:
                max_tide = month_data.loc[month_data['tide_m'].idxmax()]
                monthly_max.append({
                    'year': year,
                    'month': month,
                    'datetime': max_tide['datetime_utc'],
                    'tide_m': max_tide['tide_m']
                })
                print(f"{year}-{month:02d}       {str(max_tide['datetime_utc']):<22} "
                      f"{max_tide['tide_m']:.3f}        {max_tide['tide_m']*3.28084:.2f}")

    # Annual king tides
    print("\n" + "-" * 70)
    print("ANNUAL KING TIDES (Top 5 highest tides per year)")
    print("-" * 70)

    for year in [2023, 2024, 2025, 2026]:
        year_data = highs[highs['year'] == year].copy()
        if len(year_data) > 0:
            top5 = year_data.nlargest(5, 'tide_m')
            print(f"\n{year}:")
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                print(f"  {i}. {row['datetime_utc']}: {row['tide_m']:.3f} m ({row['tide_m']*3.28084:.2f} ft)")

    # Overall highest
    max_overall = highs.loc[highs['tide_m'].idxmax()]
    print("\n" + "-" * 70)
    print(f"HIGHEST PREDICTED TIDE (3-year period):")
    print(f"  Date/Time: {max_overall['datetime_utc']}")
    print(f"  Height: {max_overall['tide_m']:.3f} m ({max_overall['tide_m']*3.28084:.2f} ft)")
    print("-" * 70)

    # Validate January 2026 king tide
    jan2026 = highs[(highs['year'] == 2026) & (highs['month'] == 1)]
    if len(jan2026) > 0:
        jan2026_max = jan2026.loc[jan2026['tide_m'].idxmax()]
        all_time_rank = len(highs[highs['tide_m'] > jan2026_max['tide_m']]) + 1
        print(f"\nJanuary 2026 king tide validation:")
        print(f"  Peak: {jan2026_max['datetime_utc']}")
        print(f"  Height: {jan2026_max['tide_m']:.3f} m ({jan2026_max['tide_m']*3.28084:.2f} ft)")
        print(f"  Rank: #{all_time_rank} out of {len(highs)} high tides in 3-year period")

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: All high tides
    ax1 = axes[0]
    ax1.scatter(highs['datetime_utc'], highs['tide_m'], alpha=0.5, s=10, label='High tides')

    top20 = highs.nlargest(20, 'tide_m')
    ax1.scatter(top20['datetime_utc'], top20['tide_m'], c='red', s=50, marker='^',
                label='Top 20 highest', zorder=5)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('High Tide Height (m)')
    ax1.set_title('San Francisco High Tides: January 2023 - January 2026')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Monthly maximum
    ax2 = axes[1]
    monthly_df = pd.DataFrame(monthly_max)
    monthly_df['date'] = monthly_df.apply(lambda r: datetime(r['year'], r['month'], 15), axis=1)
    ax2.bar(monthly_df['date'], monthly_df['tide_m'], width=25, alpha=0.7)
    ax2.axhline(y=monthly_df['tide_m'].mean(), color='red', linestyle='--',
                label=f"Mean: {monthly_df['tide_m'].mean():.3f} m")

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Monthly Max High Tide (m)')
    ax2.set_title('Monthly Maximum High Tides')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to docs folder
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    plot_path = os.path.join(docs_dir, 'king_tides_3year_analysis.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")

    return df, highs


def plot_king_tides_forecast(model: str = None):
    """
    Generate a week-long tide forecast centered on the king tides event.
    """
    SF_LAT, SF_LON = 37.8067, -122.4650

    start = datetime(2026, 1, 1, 0, 0, 0)
    end = datetime(2026, 1, 7, 23, 59, 0)

    print("\n" + "=" * 70)
    print("KING TIDES FORECAST: January 1-7, 2026")
    print("=" * 70)
    print("\nGenerating tide forecast...")

    df = tides.predict_timeseries(
        SF_LAT, SF_LON,
        start, end,
        interval_minutes=10,
        model=model
    )

    events = tides.find_high_low_tides(df)
    highs = events[events['type'] == 'HIGH']

    print("\nHigh tides during Jan 1-7, 2026:")
    for _, row in highs.iterrows():
        print(f"  {row['datetime_utc']}: {row['tide_m']:.3f} m ({row['tide_m']*3.28084:.2f} ft)")

    max_tide = df.loc[df['tide_m'].idxmax()]
    print(f"\nPeak tide: {max_tide['datetime_utc']} at {max_tide['tide_m']:.3f} m ({max_tide['tide_m']*3.28084:.2f} ft)")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df['datetime_utc'], df['tide_m'], 'b-', linewidth=1, label='Predicted Tide')

    ax.scatter(highs['datetime_utc'], highs['tide_m'], c='red', s=50,
               marker='^', label='High Tide', zorder=5)

    lows = events[events['type'] == 'LOW']
    ax.scatter(lows['datetime_utc'], lows['tide_m'], c='green', s=50,
               marker='v', label='Low Tide', zorder=5)

    king_tide_time = datetime(2026, 1, 3, 17, 34)
    ax.axvline(king_tide_time, color='orange', linestyle='--',
               label='King Tide Peak (Jan 3)', linewidth=2)

    ax.set_xlabel('Date (UTC)', fontsize=12)
    ax.set_ylabel('Tide Elevation (m)', fontsize=12)
    ax.set_title('San Francisco Bay Tide Forecast: January 1-7, 2026\n'
                 'King Tides Event - Highest Since 1998', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    plot_path = os.path.join(docs_dir, 'king_tides_forecast_jan2026.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")


def case_study_locations(model: str = None):
    """
    Predict king tides at multiple Bay Area locations.
    """
    print("\n" + "=" * 70)
    print("KING TIDES CASE STUDY: San Francisco Bay, January 3, 2026")
    print("=" * 70)

    locations = {
        'San Francisco (Pier 41)': (37.8067, -122.4650),
        'Golden Gate Bridge': (37.8199, -122.4783),
        'Embarcadero': (37.7955, -122.3935),
        'Crissy Field': (37.8045, -122.4692),
        'Redwood City': (37.5072, -122.2116),
        'Richmond': (37.9108, -122.3567),
        'Martinez': (38.0342, -122.1253),
        'Corte Madera': (37.9250, -122.5083),
        'Larkspur': (37.9356, -122.5261),
    }

    king_tide_peak = datetime(2026, 1, 3, 17, 34, 0)

    print(f"\nPredicted astronomical tides at {king_tide_peak} UTC:")
    print("-" * 50)

    for name, (lat, lon) in locations.items():
        tide_m = tides.predict(lat, lon, king_tide_peak, model=model)
        tide_ft = tide_m * 3.28084

        print(f"{name}:")
        print(f"  Coordinates: ({lat:.4f}, {lon:.4f})")
        print(f"  Predicted tide: {tide_m:.3f} m ({tide_ft:.2f} ft)")
        print()


if __name__ == "__main__":
    # Check for EOT20 flag
    model = None
    if '--use-eot20' in sys.argv:
        eot20_path = os.path.join(tides.MODEL_DIR, 'EOT20', 'ocean_tides')
        if not os.path.exists(eot20_path):
            print("ERROR: EOT20 model not found. Download it first:")
            print("  python tides.py --download-eot20")
            sys.exit(1)
        model = 'EOT20'
        tides.set_model('EOT20')
        print("\nUsing EOT20 high-resolution model (0.125°)")

    # Run analysis
    analyze_3_year_king_tides(model)
    case_study_locations(model)
    plot_king_tides_forecast(model)

    print("\n" + "=" * 70)
    print("Analysis complete. Plots saved to docs/ folder.")
    print("=" * 70)
