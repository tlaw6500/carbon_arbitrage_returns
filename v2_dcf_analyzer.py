"""
V2 DCF Analyzer - AK Decarbonise Project
Processes BOTH Fund A (2035) and Fund B (Net Zero 2050) scenarios
"""
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple

class V2DCFAnalyzer:
    """
    DCF Analysis for both funds with three pricing scenarios each
    Fund A (2035): Investment 2025-2035, Revenue 2025-2065
    Fund B (Net Zero 2050): Investment 2025-2050, Revenue 2025-2080
    """
    
    def __init__(self):
        # LaTeX parameters
        self.discount_rate = 0.02795381840850683
        self.asset_lifetime = 30
        self.base_year = 2025
        
        # Fund configurations
        self.funds = {
            '2035': {
                'investment_end_year': 2035,
                'revenue_end_year': 2065,  # 2035 + 30 years
                'name': 'Fund A'
            },
            'Net Zero 2050': {
                'investment_end_year': 2050, 
                'revenue_end_year': 2080,  # 2050 + 30 years
                'name': 'Fund B'
            }
        }
        
        self._load_capacity_factors()
        self._load_opportunity_costs()
        
    def _load_capacity_factors(self):
        """Load capacity factors by technology"""
        cf_df = pd.read_csv('v3_capacity_weighted_average_capacity_factor.csv')
        global_row = cf_df[cf_df['region'] == 'Global'].iloc[0]
        
        self.capacity_factors = {
            'solar': global_row['Solar'],
            'onshore_wind': global_row['Wind_Onshore'], 
            'offshore_wind': global_row['Wind_Offshore'],
            'hydropower': global_row['Hydropower'],
            'geothermal': global_row['Geothermal'],
            'bioenergy': global_row['Bioenergy'] if 'Bioenergy' in global_row else 0.5
        }
        
    def _load_opportunity_costs(self):
        """Load opportunity costs - trillions USD"""
        self.opportunity_costs = pd.read_csv(
            'yearly_by_country_opportunity_cost_non_discounted_main.csv', 
            index_col=0
        )
        
    def _load_country_capacity_data(self, country_iso2: str, fund_type: str) -> Tuple[Dict, Dict, Dict]:
        """Load country capacity data for specific fund type"""
        installed_file = f'battery_yearly_installed_capacity_{fund_type}_{country_iso2}.json'
        available_file = f'battery_yearly_available_capacity_{fund_type}_{country_iso2}.json'
        unit_cost_file = f'battery_unit_ic_{country_iso2}.json'
        
        with open(installed_file, 'r') as f:
            installed_data = json.load(f)
        with open(available_file, 'r') as f:
            available_data = json.load(f)
        with open(unit_cost_file, 'r') as f:
            unit_cost_data = json.load(f)
            
        return installed_data, available_data, unit_cost_data
    
    def _get_revenue_scenarios(self, revenue_df: pd.DataFrame, country_iso2: str) -> Dict[str, Dict[str, float]]:
        """Extract three pricing scenarios from v2_data_loaders"""
        country_data = revenue_df[revenue_df['country_iso2'] == country_iso2]
        
        scenarios = {}
        for scenario in ['weighted_avg', 'min', 'max']:
            price_col = f'{scenario}_usd_mwh'
            scenario_dict = {}
            
            for _, row in country_data.iterrows():
                technology = row['Technology_Main']
                scenario_dict[technology] = row[price_col]
            
            scenarios[scenario] = scenario_dict
            
        return scenarios
    
    def calculate_fund_scenario(self, country_iso2: str, fund_type: str, revenue_scenarios: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Calculate DCF for one fund type, all pricing scenarios"""
        
        fund_config = self.funds[fund_type]
        investment_end_year = fund_config['investment_end_year']
        revenue_end_year = fund_config['revenue_end_year']
        
        # Load country data for this fund
        installed_data, available_data, unit_cost_data = self._load_country_capacity_data(country_iso2, fund_type)
        
        results = []
        
        # Process each pricing scenario
        for scenario_name, revenue_factors in revenue_scenarios.items():
            
            cash_flows = []
            
            # Calculate yearly cash flows
            for year in range(self.base_year, revenue_end_year + 1):
                year_str = str(year)
                
                # Investment (only during investment period)
                investment_cf = 0.0
                if year <= investment_end_year:
                    for technology in installed_data:
                        if technology in ['short', 'long']:
                            continue
                            
                        installed_kw = installed_data[technology].get(year_str, 0)
                        unit_cost = unit_cost_data[technology].get(year_str, 0)
                        investment_cf += installed_kw * unit_cost
                
                # Revenue (throughout asset lifetime)
                revenue_cf = 0.0
                for technology in available_data:
                    if technology in ['short', 'long']:
                        continue
                        
                    available_kw = available_data[technology].get(year_str, 0)
                    
                    if available_kw > 0 and technology in self.capacity_factors:
                        capacity_factor = self.capacity_factors[technology]
                        annual_energy_mwh = (available_kw * capacity_factor * 8760) / 1000
                        
                        if technology in revenue_factors:
                            price_per_mwh = revenue_factors[technology]
                            revenue_cf += annual_energy_mwh * price_per_mwh
                
                # Opportunity cost
                opportunity_cost_cf = 0.0
                if country_iso2 in self.opportunity_costs.index and year_str in self.opportunity_costs.columns:
                    opp_cost_trillions = self.opportunity_costs.loc[country_iso2, year_str]
                    opportunity_cost_cf = opp_cost_trillions * 1e12
                
                # Net cash flow
                net_cf = revenue_cf - investment_cf - opportunity_cost_cf
                
                # Discount factor
                discount_factor = (1 + self.discount_rate) ** (year - self.base_year)
                discounted_net_cf = net_cf / discount_factor
                
                cash_flows.append({
                    'year': year,
                    'investment_cf': investment_cf,
                    'revenue_cf': revenue_cf,
                    'opportunity_cost_cf': opportunity_cost_cf,
                    'net_cf': net_cf,
                    'discount_factor': discount_factor,
                    'discounted_net_cf': discounted_net_cf
                })
            
            # Calculate NPV
            npv = sum(cf['discounted_net_cf'] for cf in cash_flows)
            
            result = {
                'country_iso2': country_iso2,
                'fund_type': fund_type,
                'fund_name': fund_config['name'],
                'pricing_scenario': scenario_name,
                'npv_usd': npv,
                'total_investment_usd': sum(cf['investment_cf'] for cf in cash_flows),
                'total_revenue_usd': sum(cf['revenue_cf'] for cf in cash_flows),
                'total_opportunity_cost_usd': sum(cf['opportunity_cost_cf'] for cf in cash_flows),
                'investment_end_year': investment_end_year,
                'revenue_end_year': revenue_end_year,
                'cash_flows': cash_flows
            }
            
            results.append(result)
        
        return results
    
    def analyze_all_scenarios(self, revenue_table_path: str) -> List[Dict]:
        """
        Analyze ALL scenarios: 2 funds × 3 pricing × N countries
        This is the main function that produces 600+ results
        """
        revenue_df = pd.read_csv(revenue_table_path)
        all_results = []
        
        # Process each country
        for country_iso2 in revenue_df['country_iso2'].unique():
            
            # Get revenue scenarios for this country
            revenue_scenarios = self._get_revenue_scenarios(revenue_df, country_iso2)
            
            # Process BOTH fund types for this country
            for fund_type in self.funds.keys():
                try:
                    # Calculate all pricing scenarios for this fund
                    fund_results = self.calculate_fund_scenario(country_iso2, fund_type, revenue_scenarios)
                    all_results.extend(fund_results)
                    
                except FileNotFoundError:
                    # Skip if data files missing for this fund type
                    continue
        
        return all_results
    
    def export_results(self, results: List[Dict]) -> Tuple[str, str]:
        """Export NPV summary and detailed cash flows"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        
        npv_summary = []
        cash_flow_details = []
        
        for result in results:
            # NPV summary
            npv_summary.append({
                'country_iso2': result['country_iso2'],
                'fund_type': result['fund_type'],
                'fund_name': result['fund_name'],
                'pricing_scenario': result['pricing_scenario'],
                'npv_usd': result['npv_usd'],
                'total_investment_usd': result['total_investment_usd'],
                'total_revenue_usd': result['total_revenue_usd'],
                'total_opportunity_cost_usd': result['total_opportunity_cost_usd'],
                'investment_end_year': result['investment_end_year'],
                'revenue_end_year': result['revenue_end_year']
            })
            
            # Detailed cash flows
            for cf in result['cash_flows']:
                cash_flow_details.append({
                    'country_iso2': result['country_iso2'],
                    'fund_type': result['fund_type'],
                    'fund_name': result['fund_name'],
                    'pricing_scenario': result['pricing_scenario'],
                    'year': cf['year'],
                    'investment_cf': cf['investment_cf'],
                    'revenue_cf': cf['revenue_cf'],
                    'opportunity_cost_cf': cf['opportunity_cost_cf'],
                    'net_cf': cf['net_cf'],  # Undiscounted for future modules
                    'discount_factor': cf['discount_factor'],
                    'discounted_net_cf': cf['discounted_net_cf']
                })
        
        # Export files
        npv_file = f'v2_dcf_npv_summary_all_funds_{timestamp}.csv'
        cash_flow_file = f'v2_dcf_cash_flows_all_funds_{timestamp}.csv'
        
        pd.DataFrame(npv_summary).to_csv(npv_file, index=False)
        pd.DataFrame(cash_flow_details).to_csv(cash_flow_file, index=False)
        
        return npv_file, cash_flow_file

def run_v2_dcf_analysis(revenue_table_path=None) -> Tuple[List[Dict], str, str]:
    """
    Run complete V2 DCF analysis for ALL scenarios
    Output: 2 funds × 3 pricing × N countries = 600+ results
    """
    # Auto-detect v2_data_loaders output
    if not revenue_table_path:
        import glob
        files = glob.glob('v2_revenue_factor_table_*.csv')
        revenue_table_path = max(files) if files else None
        
        if not revenue_table_path:
            raise FileNotFoundError("No v2_revenue_factor_table found. Run v2_data_loaders.py first.")
    
    # Initialize and run analysis
    analyzer = V2DCFAnalyzer()
    results = analyzer.analyze_all_scenarios(revenue_table_path)
    
    # Export results
    npv_file, cash_flow_file = analyzer.export_results(results)
    
    # Summary statistics
    countries = len(set(r['country_iso2'] for r in results))
    fund_a_count = len([r for r in results if r['fund_type'] == '2035'])
    fund_b_count = len([r for r in results if r['fund_type'] == 'Net Zero 2050'])
    
    print(f"V2 DCF Analysis Complete:")
    print(f"  Countries analyzed: {countries}")
    print(f"  Fund A scenarios: {fund_a_count}")
    print(f"  Fund B scenarios: {fund_b_count}")
    print(f"  Total scenarios: {len(results)}")
    print(f"  Expected: {countries} × 2 funds × 3 pricing = {countries * 6}")
    print(f"  NPV summary: {npv_file}")
    print(f"  Cash flows: {cash_flow_file}")
    
    return results, npv_file, cash_flow_file

if __name__ == "__main__":
    # Run complete analysis for both funds
    results, npv_file, cf_file = run_v2_dcf_analysis()