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

def download_census_data():
    """
    Download demographic and economic data from Census Bureau API.
    Focus on NYC boroughs.
    """
    print("Downloading Census demographic data...")

    # Census API key - you'll need to register for one at https://www.census.gov/data/developers/data-sets.html
    api_key = None  # Set your API key here

    if not api_key:
        print("Census API key not set. Skipping Census data download.")
        return

    # ACS 5-year estimates for NYC counties (boroughs)
    counties = {
        'Bronx': '005',
        'Kings': '047',  # Brooklyn
        'New York': '061',  # Manhattan
        'Queens': '081',
        'Richmond': '085'  # Staten Island
    }

    variables = {
        'B01003_001E': 'total_population',
        'B19013_001E': 'median_household_income',
        'B25077_001E': 'median_home_value',
        'B25064_001E': 'median_gross_rent',
        'B23025_005E': 'unemployed',
        'B23025_003E': 'labor_force'
    }

    base_url = "https://api.census.gov/data/2021/acs/acs5"

    all_data = []
    for borough, county_code in counties.items():
        params = {
            'get': ','.join(variables.keys()),
            'for': f'county:{county_code}',
            'in': 'state:36',  # NY state
            'key': api_key
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1:  # Header + data
                headers = data[0]
                values = data[1]
                row = dict(zip(headers, values))
                row['borough'] = borough
                # Rename variables
                for var_code, var_name in variables.items():
                    if var_code in row:
                        row[var_name] = row.pop(var_code)
                all_data.append(row)
        else:
            print(f"Error fetching Census data for {borough}: {response.status_code}")

    if all_data:
        df = pd.DataFrame(all_data)
        filename = "nyc_census_demographics_2021.csv"
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved Census data to {filepath}")
    else:
        print("No Census data retrieved")

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