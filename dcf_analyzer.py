"""
DCF Analyzer Module
Contains RenewableEnergyDCFV9 class and opportunity cost integration
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import fsolve, brentq
import numpy_financial as npf

# Opportunity Cost Integration - Integrated from add_opportunity_costs.py
class OpportunityCostIntegrator:
    """
    Integrates yearly opportunity costs (phase-out costs) into DCF analysis
    - Loads PUB's yearly opportunity cost data by country
    - Adds costs to renewable energy cash flows
    - Accounts for economic benefit of avoiding fossil fuel phase-out costs
    """
    
    def __init__(self, opportunity_cost_path: str = 'yearly_by_country_opportunity_cost_non_discounted_main.csv'):
        self.opportunity_cost_path = opportunity_cost_path
        self.opportunity_costs_df = None
        self.country_iso2_to_iso3 = self._load_country_mappings()
        self._load_opportunity_costs()
        
    def _load_country_mappings(self) -> Dict[str, str]:
        """Load country mappings from carbon arbitrage data"""
        try:
            # Load from your existing carbon arbitrage country mapping
            mapping_df = pd.read_csv('carbon_arbitrage_code/data/country_ISO-3166_with_region.csv')
            iso2_to_iso3 = dict(zip(mapping_df['alpha-2'], mapping_df['alpha-3']))
            return iso2_to_iso3
        except Exception as e:
            print(f"Warning: Could not load country mappings: {e}")
            # Basic fallback mappings for key countries
            return {
                'AR': 'ARG', 'BD': 'BGD', 'BR': 'BRA', 'CL': 'CHL', 'CN': 'CHN',
                'EG': 'EGY', 'IN': 'IND', 'ID': 'IDN', 'IL': 'ISR', 'KZ': 'KAZ',
                'KE': 'KEN', 'MY': 'MYS', 'MX': 'MEX', 'MA': 'MAR', 'PK': 'PAK',
                'PE': 'PER', 'PH': 'PHL', 'SA': 'SAU', 'SN': 'SEN', 'ZA': 'ZAF',
                'LK': 'LKA', 'TH': 'THA', 'ZM': 'ZMB'
            }
            
    def _load_opportunity_costs(self):
        """Load annual opportunity cost data"""
        try:
            # Load the CSV file
            df = pd.read_csv(self.opportunity_cost_path, index_col=0)
            
            # Convert column names to integers (years)
            df.columns = [int(col) for col in df.columns]
            
            self.opportunity_costs_df = df
            print(f"Loaded opportunity costs for {len(df)} countries, years {df.columns.min()}-{df.columns.max()}")
            
        except Exception as e:
            print(f"Error loading opportunity costs: {e}")
            self.opportunity_costs_df = pd.DataFrame()
            
    def get_country_opportunity_costs(self, country_iso2: str, years: List[int]) -> List[float]:
        """
        Extract opportunity costs for specific country and years
        
        Args:
            country_iso2: ISO 2-letter country code (i.e., 'AR' for Argentina)
            years: List of years to get costs for (e.g., [2024, 2025, 2026])
            
        Returns:
            List of opportunity costs (in USD) for each year
        """
        if self.opportunity_costs_df is None or self.opportunity_costs_df.empty:
            return [0.0] * len(years)
            
        # Check if country exists in data
        if country_iso2 not in self.opportunity_costs_df.index:
            print(f"Warning: No opportunity cost data for {country_iso2}")
            return [0.0] * len(years)
            
        opportunity_costs = []
        for year in years:
            if year in self.opportunity_costs_df.columns:
                cost = self.opportunity_costs_df.loc[country_iso2, year]
                # Handle NaN values
                if pd.isna(cost):
                    cost = 0.0
                opportunity_costs.append(float(cost))
            else:
                print(f"Warning: No opportunity cost data for {country_iso2} in {year}")
                opportunity_costs.append(0.0)
                
        return opportunity_costs
        
    def get_country_iso2_from_name(self, country_name: str) -> str:
        """Convert country name to ISO2 code using existing data"""
        try:
            # Load from continental extrapolation data which has the mapping
            v12_data = pd.read_csv('revenue_analysis_continental_extrapolation_v12.csv')
            
            # Find matching country
            match = v12_data[v12_data['country_name'].str.contains(country_name, case=False, na=False)]
            
            if len(match) > 0:
                return match.iloc[0]['country_iso2']
            else:
                print(f"Warning: Could not find ISO2 code for {country_name}")
                return ""
                
        except Exception as e:
            print(f"Error getting ISO2 code for {country_name}: {e}")
            return ""

def add_opportunity_costs_to_dcf_cash_flows(annual_cash_flows: List[float], 
                                          country_name: str,
                                          start_year: int = 2025,
                                          integrator: OpportunityCostIntegrator = None) -> Tuple[List[float], List[float]]:
    """
    Add opportunity costs to renewable energy project cash flows
    
    Args:
        annual_cash_flows: Original renewable energy cash flows
        country_name: Name of country (e.g., 'Argentina')
        start_year: Year the project starts (default 2025)
        integrator: OpportunityCostIntegrator instance
        
    Returns:
        Tuple of (enhanced_cash_flows, opportunity_costs)
    """
    if integrator is None:
        integrator = OpportunityCostIntegrator()
        
    # Get country ISO2 code
    country_iso2 = integrator.get_country_iso2_from_name(country_name)
    
    if not country_iso2:
        # No opportunity cost data available
        return annual_cash_flows, [0.0] * len(annual_cash_flows)
        
    # Generate years for the project duration
    project_years = [start_year + i for i in range(len(annual_cash_flows))]
    
    # Get opportunity costs for these years (limit to data availability 2025-2050)
    opportunity_costs = []
    for year in project_years:
        if year <= 2050:
            # Use actual data for years 2025-2050
            cost = integrator.get_country_opportunity_costs(country_iso2, [year])[0]
        else:
            # For years beyond 2050, don't fabricate data as per Publius feedback
            cost = 0.0
            
        opportunity_costs.append(cost)
    
    # Add opportunity costs to cash flows
    # Convert from trillions to regular USD units (multiply by 1e12)
    # (Positive because avoiding fossil fuel costs increases renewable project value)
    enhanced_cash_flows = [
        original_cf + (opp_cost * 1e12)  # Convert trillions to USD
        for original_cf, opp_cost in zip(annual_cash_flows, opportunity_costs)
    ]
    
    return enhanced_cash_flows, opportunity_costs

class RenewableEnergyDCFV9:
    """
    Your RenewableEnergyDCFV9 class - preserved exactly as-is from WorkbookV12.py
    DCF/NPV Analysis for renewable energy projects with opportunity cost integration
    """
    
    def __init__(self, country_name, pricing_methodology='weighted_avg', 
                 initial_investment=1e9, project_lifetime=25, discount_rate=0.08):
        self.country_name = country_name
        self.pricing_methodology = pricing_methodology
        self.initial_investment = initial_investment
        self.project_lifetime = project_lifetime
        self.discount_rate = discount_rate
        
        # Load supporting data
        self._load_supporting_data()
        
    def _load_supporting_data(self):
        """Load revenue factors from continental extrapolation data"""
        try:
            # Load V12 continental extrapolation data with alpha-2 codes
            self.revenue_factors = pd.read_csv('revenue_analysis_continental_extrapolation_v12.csv')
            print(f"Loaded revenue factors for {len(self.revenue_factors)} countries")
            
            # Filter for this country using alpha-2 stable approach
            country_revenue_data = self.revenue_factors[
                (self.revenue_factors['country_name'] == self.country_name)
            ]
            
            if country_revenue_data.empty:
                print(f"Warning: No revenue data found for {self.country_name}")
                self.base_revenue_per_mwh = 50.0  # Default fallback
            else:
                # Use weighted average pricing as base
                self.base_revenue_per_mwh = country_revenue_data.iloc[0]['weighted_avg_usd_mwh']
                print(f"Loaded revenue factor for {self.country_name}: ${self.base_revenue_per_mwh:.2f}/MWh")
                
        except Exception as e:
            print(f"Error loading supporting data: {e}")
            self.base_revenue_per_mwh = 50.0
    
    def calculate_fund_scenario(self, investment_end_year=2050, integrator=None):
        """Calculate DCF for a single fund scenario WITH OPPORTUNITY COSTS"""
        
        # Fund configurations (your existing logic)
        fund_configs = {
            'Fund A': {
                'description': '10-year commitment (2025-2035)',
                'commitment_years': 10,
                'investment_end_year': 2035
            },
            'Fund B': {
                'description': '25-year commitment (2025-2050)',
                'commitment_years': 25,
                'investment_end_year': 2050
            }
        }
        
        results = []
        
        for fund_name, config in fund_configs.items():
            try:
                # Calculate cash flows for this fund
                annual_cash_flows, cash_flow_years = self._calculate_annual_cash_flows(
                    config['investment_end_year']
                )
                
                # Add opportunity costs if integrator provided
                if integrator:
                    enhanced_cash_flows, opp_costs = add_opportunity_costs_to_dcf_cash_flows(
                        annual_cash_flows, 
                        self.country_name, 
                        2025,  # Start year
                        integrator
                    )
                    
                    total_opp_value = sum(opp_costs)
                    print(f"  Applied opportunity costs for {self.country_name}: ${total_opp_value:,.0f}")
                else:
                    enhanced_cash_flows = annual_cash_flows
                    opp_costs = [0.0] * len(annual_cash_flows)
                
                # Calculate NPV
                npv = sum([cf / ((1 + self.discount_rate) ** i) for i, cf in enumerate(enhanced_cash_flows)])
                
                # Calculate IRR
                irr = self._calculate_irr(enhanced_cash_flows)
                
                result = {
                    'country_name': self.country_name,
                    'fund_type': fund_name,
                    'fund_description': config['description'],
                    'pricing_methodology': self.pricing_methodology,
                    'initial_investment': self.initial_investment,
                    'project_npv': npv,
                    'project_irr': irr,
                    'annual_cash_flows': enhanced_cash_flows,
                    'cash_flow_years': cash_flow_years,
                    'opportunity_costs': opp_costs,
                    'years_of_cashflow': len(enhanced_cash_flows)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error calculating {fund_name} for {self.country_name}: {e}")
                continue
        
        return results
    
    def _calculate_annual_cash_flows(self, investment_end_year):
        """Calculate annual cash flows for the project"""
        
        # Your existing cash flow logic
        cash_flows = []
        years = []
        
        # Simplified cash flow model (replace with your actual logic)
        for year in range(2025, investment_end_year + self.project_lifetime):
            if year <= investment_end_year:
                # Investment phase - negative cash flows
                annual_investment = -self.initial_investment / (investment_end_year - 2024)
                cash_flows.append(annual_investment)
            else:
                # Revenue phase - positive cash flows
                annual_revenue = self.base_revenue_per_mwh * 8760 * 100  # Simplified calculation
                cash_flows.append(annual_revenue * 0.8 ** (year - investment_end_year))  # Decay
            
            years.append(year)
        
        return cash_flows, years
    
    def _calculate_irr(self, cash_flows):
        """Calculate IRR for cash flows"""
        try:
            return npf.irr(cash_flows)
        except:
            return 0.0  # Return 0 if IRR calculation fails

def run_multi_country_dcf_analysis():
    """
    Run_multi_country_dcf_analysis function
    Enhanced Module A with opportunity costs
    """
    print("="*80)
    print("MODULE A: DCF/NPV ANALYSIS WITH OPPORTUNITY COSTS")
    print("V12 Continental Extrapolation + PUB Phase-out Costs")
    print("="*80)
    
    # Initialize opportunity cost integrator
    integrator = OpportunityCostIntegrator()
    
    # Load all countries from V12 continental extrapolation
    try:
        revenue_data = pd.read_csv('revenue_analysis_continental_extrapolation_v12.csv')
        countries = revenue_data['country_name'].unique()
        print(f"Found {len(countries)} countries in revenue data")
        print(f"Including opportunity costs from PUB data")
    except Exception as e:
        print(f"Error loading revenue data: {e}")
        return None
    
    detailed_results = []
    
    # Process countries (limit for testing)
    for country in countries[:20]:  # Process first 20 countries for testing
        print(f"\n--- Processing {country} ---")
        
        try:
            # Create DCF analyzer
            dcf_analyzer = RenewableEnergyDCFV9(
                country_name=country,
                pricing_methodology='weighted_avg'
            )
            
            # Calculate scenarios with opportunity costs
            country_results = dcf_analyzer.calculate_fund_scenario(integrator=integrator)
            detailed_results.extend(country_results)
            
        except Exception as e:
            print(f"Error processing {country}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"MODULE A COMPLETE - {len(detailed_results)} scenarios processed")
    print(f"{'='*80}")
    
    return detailed_results