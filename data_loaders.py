"""
Data Loaders Module
Contains EnhancedSimpleRevenueAnalyzer and SimpleRevenueAnalyzer classes
Handles continental extrapolation and country mapping with alpha-2 codes
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union

class EnhancedSimpleRevenueAnalyzer:
    """
    ENHANCED SimpleRevenueAnalyzer with Continental Extrapolation
    Scales from 15 countries with actual data to ALL developing countries
    using hierarchical extrapolation: Direct → Continental → Global
    """
    
    def __init__(self, csv_path='renewable_energy_revenue_table_restructured_v9.csv', 
                 data_directory='.', carbon_arbitrage_path='./carbon_arbitrage_code'):
        
        print("="*80)
        print("EMDE RENEWABLE ENERGY ANALYSIS V12 WITH CONTINENTAL EXTRAPOLATION")
        print("="*80)
        
        self.csv_path = csv_path
        self.data_directory = data_directory
        self.carbon_arbitrage_path = carbon_arbitrage_path
        
        # Load datasets with encoding handling
        try:
            self.df_raw = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.df_raw = pd.read_csv(csv_path, encoding='latin-1')
            except:
                self.df_raw = pd.read_csv(csv_path, encoding='cp1252')
        print(f"Loaded {len(self.df_raw)} raw deals from {csv_path}")
        
        # Load classification data from carbon arbitrage codebase
        self.load_classification_data()
        
        # Filter to developing countries
        self.filter_to_developing_countries()
        
        # Analyze pricing patterns
        self.analyze_pricing_patterns()
        
        # Calculate country statistics (for the 15 countries with data)
        self.country_stats_df = self.calculate_country_statistics()
        
        # CONTINENTAL EXTRAPOLATION: Expand to all developing countries
        self.extrapolated_stats_df = self.extrapolate_to_all_countries()
        
        print(f"\nCONTINENTAL EXTRAPOLATION COMPLETE:")
        print(f"Original countries with data: {len(self.country_stats_df)}")
        print(f"Total developing countries covered: {len(self.extrapolated_stats_df)}")
        print(f"Extrapolation ratio: {len(self.extrapolated_stats_df) / len(self.country_stats_df):.1f}x")
        
    def load_classification_data(self):
        """Load country classifications from carbon arbitrage codebase"""
        
        # Continental/regional mapping
        iso_path = os.path.join(self.carbon_arbitrage_path, 'data', 'country_ISO-3166_with_region.csv')
        self.iso_regions = pd.read_csv(iso_path)
        print(f"Loaded continental mapping for {len(self.iso_regions)} countries")
        
        # Development status
        unfccc_path = os.path.join(self.carbon_arbitrage_path, 'data', 'unfcc_classification_countries.csv')
        unfccc_df = pd.read_csv(unfccc_path, encoding='utf-8')
        
        # Clean the classification data
        unfccc_df = unfccc_df.dropna(subset=['asset_location', 'classification'])
        
        self.developing_countries = unfccc_df[
            unfccc_df['classification'] == 'Developing'
        ]['asset_location'].tolist()
        
        self.developed_countries = unfccc_df[
            unfccc_df['classification'] == 'All Developed'
        ]['asset_location'].tolist()
        
        print(f"Classification: {len(self.developing_countries)} developing, {len(self.developed_countries)} developed countries")
        
        # Create country name to ISO2 mapping for alpha-2 conversion
        self.country_to_iso2 = dict(zip(self.iso_regions['name'], self.iso_regions['alpha-2']))
        
    def filter_to_developing_countries(self):
        """Filter data to developing countries only"""
        
        # Convert asset location to developing country filter
        self.df_developing = self.df_raw[
            self.df_raw['asset_location'].isin(self.developing_countries)
        ].copy()
        
        print(f"Filtered to {len(self.df_developing)} deals in developing countries")
        print(f"Countries with data: {sorted(self.df_developing['asset_location'].unique())}")
        
    def analyze_pricing_patterns(self):
        """Analyze pricing patterns in the data"""
        
        # Technology analysis
        self.tech_analysis = self.df_developing.groupby('primary_technology').agg({
            'revenue_per_mwh_usd_2022': ['count', 'mean', 'median', 'std'],
            'asset_location': 'nunique'
        }).round(2)
        
        print(f"\nTechnology Analysis:")
        for tech in self.tech_analysis.index:
            row = self.tech_analysis.loc[tech]
            print(f"  {tech}: {row[('revenue_per_mwh_usd_2022', 'count')]} deals, "
                  f"{row[('asset_location', 'nunique')]} countries, "
                  f"avg ${row[('revenue_per_mwh_usd_2022', 'mean')]:,.0f}/MWh")
        
    def calculate_country_statistics(self):
        """Calculate statistics for countries with actual data"""
        
        country_stats = []
        
        for country in self.df_developing['asset_location'].unique():
            country_data = self.df_developing[self.df_developing['asset_location'] == country]
            
            # Get ISO2 code for alpha-2 conversion
            iso2_code = self.country_to_iso2.get(country, 'XX')
            
            # Get continent
            continent_row = self.iso_regions[self.iso_regions['name'] == country]
            continent = continent_row['region'].iloc[0] if len(continent_row) > 0 else 'Unknown'
            
            stats = {
                'country_name': country,
                'country_iso2': iso2_code,  # Alpha-2 code as requested by Publius
                'continent': continent,
                'deal_count': len(country_data),
                'weighted_avg_usd_mwh': np.average(
                    country_data['revenue_per_mwh_usd_2022'], 
                    weights=country_data['installed_capacity_mw']
                ),
                'min_usd_mwh': country_data['revenue_per_mwh_usd_2022'].min(),
                'max_usd_mwh': country_data['revenue_per_mwh_usd_2022'].max(),
                'total_capacity_mw': country_data['installed_capacity_mw'].sum(),
                'technologies': list(country_data['primary_technology'].unique()),
                'data_source': 'direct'
            }
            country_stats.append(stats)
        
        return pd.DataFrame(country_stats)
        
    def extrapolate_to_all_countries(self):
        """
        CONTINENTAL EXTRAPOLATION
        Expand from 15 countries to all developing countries using continental averages
        """
        
        # Calculate continental averages from countries with data
        continental_averages = self.country_stats_df.groupby('continent').agg({
            'weighted_avg_usd_mwh': 'mean',
            'min_usd_mwh': 'mean', 
            'max_usd_mwh': 'mean'
        }).round(2)
        
        print(f"\nContinental Averages:")
        for continent in continental_averages.index:
            avg = continental_averages.loc[continent, 'weighted_avg_usd_mwh']
            print(f"  {continent}: ${avg:.2f}/MWh")
        
        # Calculate global average as fallback
        global_avg = {
            'weighted_avg_usd_mwh': self.country_stats_df['weighted_avg_usd_mwh'].mean(),
            'min_usd_mwh': self.country_stats_df['min_usd_mwh'].mean(),
            'max_usd_mwh': self.country_stats_df['max_usd_mwh'].mean()
        }
        
        # Start with countries that have direct data
        extrapolated_stats = self.country_stats_df.copy()
        
        # Add all other developing countries using continental extrapolation
        for country in self.developing_countries:
            if country not in self.country_stats_df['country_name'].values:
                
                # Get country metadata
                iso2_code = self.country_to_iso2.get(country, 'XX')
                continent_row = self.iso_regions[self.iso_regions['name'] == country]
                continent = continent_row['region'].iloc[0] if len(continent_row) > 0 else 'Unknown'
                
                # Use continental average if available, otherwise global average
                if continent in continental_averages.index:
                    pricing = continental_averages.loc[continent]
                    data_source = 'continental'
                    source_countries = list(self.country_stats_df[
                        self.country_stats_df['continent'] == continent
                    ]['country_name'])
                else:
                    pricing = global_avg
                    data_source = 'global'
                    source_countries = list(self.country_stats_df['country_name'])
                
                extrapolated_country = {
                    'country_name': country,
                    'country_iso2': iso2_code,  # Alpha-2 as requested
                    'continent': continent,
                    'deal_count': 0,  # No direct deals
                    'weighted_avg_usd_mwh': pricing['weighted_avg_usd_mwh'],
                    'min_usd_mwh': pricing['min_usd_mwh'],
                    'max_usd_mwh': pricing['max_usd_mwh'],
                    'total_capacity_mw': 0,
                    'technologies': [],
                    'data_source': data_source,
                    'source_countries': source_countries
                }
                
                extrapolated_stats = pd.concat([
                    extrapolated_stats, 
                    pd.DataFrame([extrapolated_country])
                ], ignore_index=True)
        
        # Save the extrapolated results
        output_path = 'revenue_analysis_continental_extrapolation_v12.csv'
        extrapolated_stats.to_csv(output_path, index=False)
        print(f"\nSaved continental extrapolation results: {output_path}")
        
        return extrapolated_stats

class SimpleRevenueAnalyzer:
    """
    Original SimpleRevenueAnalyzer - processes only raw deals, no extrapolation
    Kept for backward compatibility and comparison
    """
    
    def __init__(self, csv_path='renewable_energy_revenue_table_restructured_v9.csv'):
        print("="*80)
        print("SIMPLE REVENUE ANALYZER - RAW DEALS ONLY")
        print("="*80)
        
        # Load data
        try:
            self.df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.df = pd.read_csv(csv_path, encoding='latin-1')
            except:
                self.df = pd.read_csv(csv_path, encoding='cp1252')
        
        print(f"Loaded {len(self.df)} deals from {csv_path}")
        
        # Basic analysis
        self.analyze_data()
        
    def analyze_data(self):
        """Basic analysis of raw deals data"""
        print(f"Countries: {self.df['asset_location'].nunique()}")
        print(f"Technologies: {list(self.df['primary_technology'].unique())}")
        print(f"Revenue range: ${self.df['revenue_per_mwh_usd_2022'].min():.0f} - ${self.df['revenue_per_mwh_usd_2022'].max():.0f}/MWh")
        
def run_simple_revenue_analysis():
    """Run the simple revenue analysis pipeline"""
    print("\n=== SIMPLE REVENUE ANALYSIS ===")
    analyzer = SimpleRevenueAnalyzer()
    return analyzer

def run_enhanced_revenue_analysis():
    """Run the enhanced revenue analysis with continental extrapolation"""
    print("\n=== ENHANCED REVENUE ANALYSIS WITH CONTINENTAL EXTRAPOLATION ===")
    analyzer = EnhancedSimpleRevenueAnalyzer()
    return analyzer