#!/usr/bin/env python3
"""
Housing Data Processor for FinRL Housing project.

Processes raw housing and economic data into features for RL environment.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class HousingDataProcessor:
    """
    Processes housing data similar to FinRL's DataProcessor.

    Handles data cleaning, feature engineering, and array conversion.
    """

    def __init__(self, data_dir='data'):
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.indicators_dir = os.path.join(data_dir, 'indicators')
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.indicators_dir, exist_ok=True)

    def load_nyc_evictions(self, filename='nyc_evictions_2017_2023.csv'):
        """Load and process NYC eviction data."""
        filepath = os.path.join(self.raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"Eviction data file not found: {filepath}")
            return None

        df = pd.read_csv(filepath)

        # Convert executed_date to datetime
        df['executed_date'] = pd.to_datetime(df['executed_date'], errors='coerce')

        # Extract year and month
        df['year'] = df['executed_date'].dt.year
        df['month'] = df['executed_date'].dt.month

        # Group by year and borough to get eviction counts
        eviction_counts = df.groupby(['year', 'borough']).size().reset_index(name='eviction_count')

        # Pivot to have boroughs as columns
        eviction_pivot = eviction_counts.pivot_table(
            index='year',
            columns='borough',
            values='eviction_count',
            fill_value=0
        ).reset_index()

        # Rename columns
        borough_cols = [col for col in eviction_pivot.columns if col != 'year']
        eviction_pivot = eviction_pivot.rename(columns={col: f'evictions_{col.lower()}' for col in borough_cols})

        return eviction_pivot

    def generate_synthetic_economic_data(self, start_year=2010, end_year=2023):
        """
        Generate synthetic economic data since Census API requires key.
        In production, replace with real Census data.
        """
        np.random.seed(42)  # For reproducibility

        years = range(start_year, end_year + 1)
        data = []

        # Base values for NYC
        base_population = 8_300_000
        base_income = 75_000
        base_rent = 2_800
        base_unemployment = 0.045

        for year in years:
            # Add some trends and noise
            year_factor = (year - 2010) * 0.02  # Slight growth

            row = {
                'year': year,
                'total_population': base_population * (1 + year_factor + np.random.normal(0, 0.01)),
                'median_household_income': base_income * (1 + year_factor + np.random.normal(0, 0.05)),
                'median_gross_rent': base_rent * (1 + year_factor + np.random.normal(0, 0.03)),
                'unemployment_rate': max(0.02, base_unemployment + year_factor * 0.5 + np.random.normal(0, 0.005)),
                'cpi': 100 * (1 + year_factor * 0.8 + np.random.normal(0, 0.02))  # Consumer Price Index
            }
            data.append(row)

        df = pd.DataFrame(data)
        return df

    def add_affordability_index(self, df):
        """Add housing affordability metrics."""
        # Affordability ratio: % of income spent on housing (should be <30% for affordable)
        df['rent_income_ratio'] = (df['median_gross_rent'] * 12) / df['median_household_income']

        # Affordability index: inverse of ratio (higher is more affordable)
        df['affordability_index'] = 1 / df['rent_income_ratio']

        return df

    def add_displacement_risk(self, df):
        """
        Add displacement risk indicators based on rent changes and income.
        Simplified version - in reality would use gentrification metrics.
        """
        # Rent growth rate
        df['rent_growth_rate'] = df['median_gross_rent'].pct_change()

        # Displacement risk: high rent growth + low affordability
        df['displacement_risk'] = (
            df['rent_growth_rate'].fillna(0) * 0.5 +
            (1 - df['affordability_index']) * 0.5
        )

        return df

    def add_equity_metrics(self, df):
        """
        Add equity indicators (simplified).
        Real implementation would include racial/income disparity data.
        """
        # For now, use unemployment as proxy for equity issues
        df['equity_index'] = 1 / (1 + df['unemployment_rate'])

        return df

    def load_census_data(self, filename='nyc_census_demographics_2021.csv'):
        """Load Census ACS borough-level data and create a time series for integration."""
        filepath = os.path.join(self.raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"Census data file not found: {filepath}")
            return None

        df = pd.read_csv(filepath)

        # if year missing, default to 2021
        if 'year' not in df.columns:
            df['year'] = 2021

        # Numeric conversions
        for col in ['total_population', 'median_household_income', 'median_gross_rent', 'median_home_value', 'unemployed', 'labor_force']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Aggregate borough data to NYC-level by weighted population where relevant
        agg = {}
        agg['total_population'] = df['total_population'].sum()
        agg['median_household_income'] = (df['median_household_income'] * df['total_population']).sum() / agg['total_population']
        agg['median_gross_rent'] = (df['median_gross_rent'] * df['total_population']).sum() / agg['total_population']
        agg['median_home_value'] = (df['median_home_value'] * df['total_population']).sum() / agg['total_population']

        if df['labor_force'].sum() > 0:
            agg['unemployment_rate'] = df['unemployed'].sum() / df['labor_force'].sum()
        else:
            agg['unemployment_rate'] = 0.0

        # CPI: use provided CPI if available, otherwise an approximate BLS trend (210+)
        if 'cpi' in df.columns:
            df['cpi'] = pd.to_numeric(df['cpi'], errors='coerce')
            agg['cpi'] = (df['cpi'] * df['total_population']).sum() / agg['total_population']
        else:
            agg['cpi'] = 218.056  # 2010 CPI reference (approx. BLS all-items)

        # Build synthetic time series from 2010-2024 from single year values
        years = range(2010, 2024)
        data = []
        for y in years:
            # Add a simple CPI trend from BLS (approx 2.5% annual inflation)
            cpi_value = 218.056 * ((1 + 0.025) ** (y - 2010))
            row = {
                'year': y,
                'total_population': agg['total_population'],
                'median_household_income': agg['median_household_income'],
                'median_gross_rent': agg['median_gross_rent'],
                'median_home_value': agg['median_home_value'],
                'unemployment_rate': agg['unemployment_rate'],
                'cpi': cpi_value,
            }
            data.append(row)

        return pd.DataFrame(data)

    def process_data(self):
        """Main data processing pipeline."""
        print("Processing housing data...")

        # Load eviction data
        evictions = self.load_nyc_evictions()
        if evictions is None:
            print("Using synthetic eviction data...")
            years = range(2010, 2024)
            evictions = pd.DataFrame({'year': years})
            for borough in ['BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND']:
                base = {'BRONX': 8000, 'BROOKLYN': 12000, 'MANHATTAN': 10000, 'QUEENS': 9000, 'STATEN ISLAND': 2000}[borough]
                evictions[f'evictions_{borough.lower()}'] = [
                    int(base * (1 + (y-2010)*0.05 + np.random.normal(0, 0.1))) for y in years
                ]

        # Load Census or synthetic economic data
        census_data = self.load_census_data()
        if census_data is not None:
            print("Using Census ACS data for economic features")
            economic = census_data
        else:
            print("Using synthetic economic data")
            economic = self.generate_synthetic_economic_data()

        # Merge datasets
        df = pd.merge(economic, evictions, on='year', how='left').fillna(0)

        # Calculate total evictions
        eviction_cols = [col for col in df.columns if col.startswith('evictions_')]
        df['total_evictions'] = df[eviction_cols].sum(axis=1)

        # Calculate eviction rate (per 1000 people)
        df['eviction_rate'] = (df['total_evictions'] / df['total_population']) * 1000

        # Add features
        df = self.add_affordability_index(df)
        df = self.add_displacement_risk(df)
        df = self.add_equity_metrics(df)

        # Save processed data
        processed_file = os.path.join(self.processed_dir, 'housing_economic_data_processed.csv')
        df.to_csv(processed_file, index=False)
        print(f"Saved processed data to {processed_file}")

        return df

    def df_to_array(self, df, feature_cols=None):
        """
        Convert dataframe to arrays for RL environment.
        Similar to FinRL's df_to_array.
        """
        if feature_cols is None:
            # Default features for state space
            feature_cols = [
                'eviction_rate', 'median_gross_rent', 'cpi',
                'unemployment_rate', 'affordability_index',
                'displacement_risk', 'equity_index'
            ]

        # Ensure all columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) != len(feature_cols):
            print(f"Warning: Missing columns: {set(feature_cols) - set(available_cols)}")

        # Convert to numpy array
        feature_array = df[available_cols].values

        # Normalize features
        scaler = StandardScaler()
        feature_array = scaler.fit_transform(feature_array)

        # For housing RL, we might have separate arrays like FinRL
        # But for simplicity, return single array
        return feature_array

def main():
    processor = HousingDataProcessor()
    df = processor.process_data()

    # Example: convert to array
    feature_array = processor.df_to_array(df)
    print(f"Feature array shape: {feature_array.shape}")

    # Save array for later use
    np.save(os.path.join(processor.processed_dir, 'feature_array.npy'), feature_array)

if __name__ == "__main__":
    main()