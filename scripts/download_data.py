#!/usr/bin/env python3
"""
Data download script for Housing Budget Allocation project.

Fetches historical housing and economic data from:
- NYC Open Data (evictions, housing costs)
- U.S. Census Bureau (demographics, income)
- HUD (housing assistance data)
"""

import requests
import pandas as pd
import os
from datetime import datetime
import json

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
NYC_APP_TOKEN = None  # Add your NYC Open Data app token if needed for higher limits

def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)

def download_nyc_evictions(start_year=2017, end_year=2023):
    """
    Download NYC eviction data from Open Data portal.
    Data available from 2017 onwards.
    """
    print("Downloading NYC eviction data...")

    # NYC Open Data API endpoint for evictions
    base_url = "https://data.cityofnewyork.us/resource/6z8x-wfk4.json"

    all_data = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching data for {year}...")
        params = {
            '$where': f"executed_date >= '{year}-01-01T00:00:00.000' AND executed_date < '{year+1}-01-01T00:00:00.000'",
            '$limit': 50000  # Adjust based on data size
        }
        if NYC_APP_TOKEN:
            params['$$app_token'] = NYC_APP_TOKEN

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            all_data.extend(data)
            print(f"Retrieved {len(data)} records for {year}")
        else:
            print(f"Error fetching data for {year}: {response.status_code}")
            print(f"Response: {response.text[:200]}")  # Debug

    if all_data:
        df = pd.DataFrame(all_data)
        filename = f"nyc_evictions_{start_year}_{end_year}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} eviction records to {filepath}")
    else:
        print("No eviction data retrieved")

def download_census_data(year=2021):
    """
    Download demographic and economic data from Census Bureau API.
    Focus on NYC boroughs.
    """
    print(f"Downloading Census demographic data for ACS {year} 5-year...")

    # Census API key (set as env var for security) or hardcode if local only
    api_key = os.getenv("CENSUS_API_KEY", "a73922599e72700c29d229a244b5689c87a079da")

    if not api_key:
        print("Census API key not set. Skipping Census data download.")
        return

    variables = {
        'B01003_001E': 'total_population',
        'B19013_001E': 'median_household_income',
        'B25077_001E': 'median_home_value',
        'B25064_001E': 'median_gross_rent',
        'B23025_005E': 'unemployed',
        'B23025_003E': 'labor_force'
    }

    base_url = f"https://api.census.gov/data/{year}/acs/acs5"

    params = {
        'get': ','.join(variables.keys()),
        'for': 'county:005,047,061,081,085',
        'in': 'state:36',  # NY state
        'key': api_key
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Error fetching Census data: {response.status_code}")
        print(response.text[:500])
        return

    data = response.json()
    if len(data) <= 1:
        print("No Census data retrieved")
        return

    # Format to DataFrame
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    df = df.rename(columns=variables)
    df['borough'] = df['county'].map({
        '005': 'Bronx',
        '047': 'Kings',
        '061': 'New York',
        '081': 'Queens',
        '085': 'Richmond'
    })
    df['year'] = year

    # Cast numeric fields
    for col in ['total_population', 'median_household_income', 'median_home_value', 'median_gross_rent', 'unemployed', 'labor_force']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    filename = f"nyc_census_demographics_{year}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved Census data to {filepath}")


def download_hud_data():
    """
    Download housing assistance data from HUD.
    This is a simplified version - HUD data often requires manual download.
    """
    print("HUD data download requires manual access to HUD User datasets.")
    print("Please visit https://www.huduser.gov/portal/datasets.html")
    print("Key datasets to download:")
    print("- Picture of Subsidized Households")
    print("- Rental Assistance Administrative Data")
    print("- Comprehensive Housing Affordability Strategy (CHAS) data")

    # For now, create a placeholder
    placeholder_data = {
        'dataset': ['Picture of Subsidized Households', 'CHAS Data'],
        'url': ['https://www.huduser.gov/portal/datasets/picture.html',
                'https://www.huduser.gov/portal/datasets/CHAS.html'],
        'note': ['Manual download required', 'Manual download required']
    }
    df = pd.DataFrame(placeholder_data)
    filepath = os.path.join(DATA_DIR, 'hud_datasets_info.csv')
    df.to_csv(filepath, index=False)
    print(f"Created HUD data info file at {filepath}")

def main():
    ensure_data_dir()

    print("Starting data download process...")
    print(f"Data will be saved to: {DATA_DIR}")

    # Download datasets
    download_nyc_evictions()
    download_census_data()
    download_hud_data()

    print("Data download complete!")

if __name__ == "__main__":
    main()