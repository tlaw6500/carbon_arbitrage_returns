"""
IRR Calculator Module
Contains ProjectIRRCalculator class -  Module B logic 
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import numpy_financial as npf

class ProjectIRRCalculator:
    """
    ProjectIRRCalculator class - Idea from WorkbookV12.py
    Module B: Project-level IRR calculations
    """
    
    def __init__(self, module_a_results: list):
        """Initialize with Module A results"""
        self.module_a_results = module_a_results
        
        # Country codes mapping for alpha-2 conversion (per Publius request)
        self.country_codes = {
            'Argentina': 'AR', 'Brazil': 'BR', 'Chile': 'CL', 'India': 'IN', 'Indonesia': 'ID',
            'Bangladesh': 'BD', 'Philippines': 'PH', 'Thailand': 'TH', 'Malaysia': 'MY',
            'South Africa': 'ZA', 'Egypt': 'EG', 'Morocco': 'MA', 'Kenya': 'KE',
            'Kazakhstan': 'KZ', 'Pakistan': 'PK', 'Peru': 'PE', 'Colombia': 'CO',
            'China': 'CN', 'Mexico': 'MX', 'Turkey': 'TR', 'UAE': 'AE',
            'Myanmar': 'MM', 'Nigeria': 'NG', 'Senegal': 'SN', 'Sri Lanka': 'LK',
            'Taiwan': 'TW', 'Zambia': 'ZM', 'Saudi Arabia': 'SA'
        }
        
        print(f"Initialized IRR Calculator with {len(module_a_results)} scenarios")
    
    def calculate_project_irr(self, cash_flows: list, investment: float) -> dict:
        """Calculate IRR for a project given cash flows and initial investment"""
        
        try:
            # Ensure cash flows start with negative investment
            if cash_flows[0] > 0:
                cash_flows = [-investment] + cash_flows
            
            # Calculate IRR using numpy_financial
            irr = npf.irr(cash_flows)
            
            # Handle NaN results
            if pd.isna(irr):
                return {'irr': 0.0, 'status': 'NO_SOLUTION'}
            
            return {
                'irr': float(irr),
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            return {
                'irr': 0.0, 
                'status': 'ERROR',
                'error': str(e)
            }
    
    def process_all_scenarios(self):
        """Process ALL countries"""
        
        irr_results = []
        
        for result in self.module_a_results:
            # Extract data using ISO-2 as primary identifier
            country_iso2 = result.get('country_iso2', result.get('country_name', ''))  # Fallback for transition
            pricing_method = result['pricing_methodology']
            fund_type = result['fund_type']
            
            # Use ISO-2 directly - no conversion needed
            if not country_iso2:
                print(f"Warning: No country ISO-2 code found for result: {result}")
                continue
            
            # Get financial data
            cash_flows = result.get('annual_cash_flows', [])
            initial_investment = result.get('initial_investment', 0)
            project_npv = result.get('project_npv', 0)
            
            # Calculate IRR
            irr_calculation = self.calculate_project_irr(cash_flows, initial_investment)
            
            # Create result record
            irr_result = {
                'country_iso2': country_iso2,  # Use ISO-2 as primary identifier
                'fund_type': fund_type,
                'pricing_methodology': pricing_method,
                'project_irr': irr_calculation['irr'],
                'irr_status': irr_calculation['status'],
                'initial_investment': initial_investment,
                'annual_cash_flows': cash_flows,
                'cash_flow_years': result.get('cash_flow_years', []),
                'project_npv': project_npv,
                'years_of_cashflow': len(cash_flows)
            }
            
            # Add fund description if available
            if 'fund_description' in result:
                irr_result['fund_description'] = result['fund_description']
            
            irr_results.append(irr_result)
        
        return irr_results
    
    def create_complete_results_table(self) -> pd.DataFrame:
        """Create complete results table for all scenarios"""
        
        # Process all scenarios
        irr_results = self.process_all_scenarios()
        
        # Convert to DataFrame
        df = pd.DataFrame(irr_results)
        
        # Format IRR as percentage for display
        df['IRR_Percent'] = (df['project_irr'] * 100).round(2)
        
        # Convert investment to billions
        df['Investment_B'] = (df['initial_investment'] / 1e9).round(2)
        
        # Convert NPV to billions  
        df['NPV_B'] = (df['project_npv'] / 1e9).round(2)
        
        # Select display columns
        display_cols = [
            'country_name', 'country_code', 'fund_type', 'pricing_methodology',
            'IRR_Percent', 'Investment_B', 'NPV_B', 'years_of_cashflow', 'irr_status'
        ]
        
        return df[display_cols]
    
    def generate_summary_statistics(self, irr_results: list) -> dict:
        """Generate summary statistics from IRR results"""
        
        # Filter successful calculations
        successful_irrs = [
            r['project_irr'] for r in irr_results 
            if r['irr_status'] == 'SUCCESS' and not pd.isna(r['project_irr'])
        ]
        
        if not successful_irrs:
            return {
                'total_scenarios': len(irr_results),
                'successful': 0,
                'failed': len(irr_results),
                'mean_irr': 0,
                'median_irr': 0,
                'irr_range': (0, 0)
            }
        
        # Calculate statistics
        stats = {
            'total_scenarios': len(irr_results),
            'successful': len(successful_irrs),
            'failed': len(irr_results) - len(successful_irrs),
            'mean_irr': np.mean(successful_irrs) * 100,  # Convert to percentage
            'median_irr': np.median(successful_irrs) * 100,
            'irr_range': (np.min(successful_irrs) * 100, np.max(successful_irrs) * 100)
        }
        
        return stats

def run_module_b_irr_analysis(module_a_results):
    """
    Your run_module_b_irr_analysis function - preserved exactly as-is
    Module B: Calculate IRR for all Fund/Country/Pricing combinations
    """
    print("="*80)
    print("MODULE B: PROJECT-LEVEL IRR ANALYSIS")
    print("Calculates IRR for all Fund/Country/Pricing combinations")
    print("="*80)

    irr_calculator = ProjectIRRCalculator(module_a_results)
    irr_results = irr_calculator.process_all_scenarios()

    # Display COMPLETE results table
    complete_table = irr_calculator.create_complete_results_table()
    
    print(f"\nCOMPLETE IRR RESULTS TABLE ({len(complete_table)} scenarios):")
    print("="*120)
    print(complete_table.to_string(index=False))

    # Generate summary statistics  
    summary_stats = irr_calculator.generate_summary_statistics(irr_results)
    
    print("\n" + "="*70)
    print("MODULE B SUMMARY STATISTICS:")
    print("="*70)
    print(f"Total scenarios processed: {summary_stats['total_scenarios']}")
    print(f"Successful IRR calculations: {summary_stats['successful']} ({summary_stats['successful']/summary_stats['total_scenarios']*100:.1f}%)")
    print(f"Failed calculations: {summary_stats['failed']} ({summary_stats['failed']/summary_stats['total_scenarios']*100:.1f}%)")
    
    if summary_stats['successful'] > 0:
        print(f"IRR Statistics for successful calculations:")
        print(f"  Mean IRR: {summary_stats['mean_irr']:.2f}%")
        print(f"  Median IRR: {summary_stats['median_irr']:.2f}%")
        print(f"  IRR Range: {summary_stats['irr_range'][0]:.2f}% to {summary_stats['irr_range'][1]:.2f}%")
    
    # Failure breakdown
    failure_counts = {}
    for result in irr_results:
        if result['irr_status'] != 'SUCCESS':
            status = result['irr_status']
            failure_counts[status] = failure_counts.get(status, 0) + 1
    
    if failure_counts:
        print(f"\nFailure breakdown:")
        for status, count in failure_counts.items():
            print(f"  {status}: {count} scenarios")

    print(f"\nModule B Complete! Ready for Module C (Waterfall Analysis)")
    print(f"Output: {len(irr_results)} detailed IRR results")
    
    return irr_results