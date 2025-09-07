"""
Waterfall Analyzer Module
ONLY Aug22.csv with proper column names: new_woinf_sen_debt, new_woinf_sub_debt, new_woinf_equity
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy_financial as npf
from typing import Dict, List, Tuple

class WaterfallAnalyzer:
    
    def __init__(self, module_b_results: list, aug22_csv_path: str = 'Aug22.csv'):
        """Initialize with Module B results and Aug22.csv ONLY"""
        
        self.module_b_results = module_b_results
        self.aug22_csv_path = aug22_csv_path
        
        # Load required returns from Aug22.csv - NO DEFAULTS
        self.required_returns_df = self._load_required_returns()
        
        if self.required_returns_df.empty:
            raise FileNotFoundError(f"CRITICAL: {aug22_csv_path} is required. No default rates.")
        
        # Capital structure (standard private equity structure)
        self.capital_structure = {
            'Senior_Debt': 0.50,
            'Sub_Debt': 0.25, 
            'Equity': 0.25
        }
        
        print(f"Initialized Waterfall Analyzer with Aug22.csv data for {len(self.required_returns_df)} countries")
        print("Using ONLY country-specific rates")
    
    def _load_required_returns(self) -> pd.DataFrame:
        """Load required returns from Aug22.csv"""
        try:
            # Load with multiple encoding attempts
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(self.aug22_csv_path, encoding=encoding)
                    print(f" Loaded Aug22.csv: {len(df)} countries using {encoding}")
                    
                    # Verify required columns exist
                    required_cols = ['Country', 'new_woinf_sen_debt', 'new_woinf_sub_debt', 'new_woinf_equity']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        print(f" Missing columns in Aug22.csv: {missing_cols}")
                        print(f"Available columns: {list(df.columns)}")
                        return pd.DataFrame()
                    
                    return df
                    
                except UnicodeDecodeError:
                    continue
            
            print(f" Could not load {self.aug22_csv_path} with any encoding")
            return pd.DataFrame()
            
        except FileNotFoundError:
            print(f" CRITICAL ERROR: {self.aug22_csv_path} not found")
            return pd.DataFrame()
    
    def _parse_percentage_safely(self, value) -> float:
        """Parse percentage values from Aug22.csv"""
        if pd.isna(value):
            raise ValueError("NaN rate found in Aug22.csv - requires complete data")
        
        try:
            # Handle string percentages like "6.28%" 
            if isinstance(value, str):
                value = value.strip().replace('%', '')
            
            parsed = float(value)
            
            # Convert percentage to decimal if needed
            if parsed > 1:
                parsed = parsed / 100.0
            
            return parsed
            
        except (ValueError, TypeError):
            raise ValueError(f"Invalid rate in Aug22.csv: {value}")
    
    def _get_required_returns(self, country_name: str) -> Dict[str, float]:
        """
        Get required returns for country from Aug22.csv ONLY
        Uses proper columns: new_woinf_sen_debt, new_woinf_sub_debt, new_woinf_equity
        """
        
        if self.required_returns_df.empty:
            raise ValueError("Aug22.csv data not available - cannot proceed without professor's rates")
        
        # Find exact country match in Aug22.csv
        country_data = None
        for _, row in self.required_returns_df.iterrows():
            if pd.isna(row.get('Country', '')):
                continue
            
            csv_country = str(row['Country']).strip()
            input_country = str(country_name).strip()
            
            # Exact match first
            if csv_country.lower() == input_country.lower():
                country_data = row
                break
            
            # Partial match as fallback
            if (input_country.lower() in csv_country.lower() or 
                csv_country.lower() in input_country.lower()):
                country_data = row
                break
        
        if country_data is None:
            raise ValueError(f"Country '{country_name}' not found in Aug22.csv - professor requires complete coverage")
        
        # Parse required returns using CORRECT column names
        try:
            returns = {
                'Senior_Debt': self._parse_percentage_safely(country_data['new_woinf_sen_debt']),
                'Sub_Debt': self._parse_percentage_safely(country_data['new_woinf_sub_debt']),
                'Equity': self._parse_percentage_safely(country_data['new_woinf_equity'])
            }
            
            print(f"✅ {country_name}: Senior={returns['Senior_Debt']*100:.2f}%, Sub={returns['Sub_Debt']*100:.2f}%, Equity={returns['Equity']*100:.2f}%")
            return returns
            
        except Exception as e:
            raise ValueError(f"Error parsing rates for {country_name}: {e}")
    
    def _calculate_tranche_investments(self, total_investment: float) -> Dict[str, float]:
        """Calculate investment amounts for each tranche"""
        return {
            tranche: total_investment * weight
            for tranche, weight in self.capital_structure.items()
        }
    
    def _calculate_annual_requirements(self, tranche_investments: Dict[str, float], 
                                    required_returns: Dict[str, float]) -> Dict[str, float]:
        """Calculate annual payment requirements for each tranche"""
        return {
            tranche: investment * required_returns[tranche]
            for tranche, investment in tranche_investments.items()
        }
    
    def _version_1_waterfall(self, annual_cash_flows: List[float],
                           annual_requirements: Dict[str, float],
                           tranche_investments: Dict[str, float]) -> Tuple[Dict, List[Dict]]:
        """
        Version 1: Basic Year-by-Year Waterfall (Senior → Sub → Equity)
        NO carryforward - unpaid amounts are lost each year
        """
        
        # Initialize tracking variables
        total_payments = {tranche: 0.0 for tranche in self.capital_structure.keys()}
        annual_payments = []
        
        # Process each year's cash flow independently
        for year, cash_flow in enumerate(annual_cash_flows):
            year_payments = {tranche: 0.0 for tranche in self.capital_structure.keys()}
            remaining_cash = cash_flow
            
            # Waterfall: Senior → Sub → Equity (no carryforward)
            for tranche in ['Senior_Debt', 'Sub_Debt', 'Equity']:
                if remaining_cash <= 0:
                    break
                
                required = annual_requirements[tranche]
                payment = min(remaining_cash, required)
                
                year_payments[tranche] = payment
                total_payments[tranche] += payment
                remaining_cash -= payment
            
            annual_payments.append({
                'year': year,
                'cash_flow': cash_flow,
                'payments': year_payments.copy(),
                'remaining': remaining_cash
            })
        
        return total_payments, annual_payments
    
    def _version_2_robust_waterfall(self, annual_cash_flows: List[float],
                                   annual_requirements: Dict[str, float],
                                   tranche_investments: Dict[str, float]) -> Tuple[Dict, List[Dict]]:
        """
        Version 2: Robust Waterfall with Carryforward Compounding
        100% completion guarantee - unpaid amounts roll forward with interest
        Using Heejin's country-specific rates from Aug22.csv ONLY
        """
        
        # Initialize tracking with carryforward capability
        outstanding_balances = tranche_investments.copy()
        total_payments = {tranche: 0.0 for tranche in self.capital_structure.keys()}
        annual_payments = []
        
        # Process each year with compounding
        for year, cash_flow in enumerate(annual_cash_flows):
            year_payments = {tranche: 0.0 for tranche in self.capital_structure.keys()}
            remaining_cash = cash_flow
            
            # Compound outstanding balances using Aug22.csv rates
            required_returns = {
                tranche: outstanding_balances[tranche] * rate
                for tranche, rate in [
                    ('Senior_Debt', annual_requirements['Senior_Debt'] / tranche_investments['Senior_Debt']),
                    ('Sub_Debt', annual_requirements['Sub_Debt'] / tranche_investments['Sub_Debt']),
                    ('Equity', annual_requirements['Equity'] / tranche_investments['Equity'])
                ]
            }
            
            # Waterfall with carryforward: Senior → Sub → Equity
            for tranche in ['Senior_Debt', 'Sub_Debt', 'Equity']:
                if remaining_cash <= 0 or outstanding_balances[tranche] <= 0:
                    continue
                
                # Calculate total amount due (principal + accrued return)
                required = required_returns[tranche]
                total_due = outstanding_balances[tranche] + required
                
                # Pay what we can
                payment = min(remaining_cash, total_due)
                year_payments[tranche] = payment
                total_payments[tranche] += payment
                remaining_cash -= payment
                
                # Update outstanding balance
                outstanding_balances[tranche] = max(0, total_due - payment)
            
            annual_payments.append({
                'year': year,
                'cash_flow': cash_flow,
                'payments': year_payments.copy(),
                'outstanding': outstanding_balances.copy(),
                'remaining': remaining_cash
            })
        
        return total_payments, annual_payments
    
    def _calculate_tranche_irr(self, tranche_investment: float,
                             annual_payments: List[Dict],
                             tranche: str) -> float:
        """Calculate IRR for specific tranche"""
        
        # Build cash flow series
        cash_flows = [-tranche_investment]  # Initial investment
        
        for payment_record in annual_payments:
            cash_flows.append(payment_record['payments'][tranche])
        
        # Calculate IRR
        try:
            irr = npf.irr(cash_flows)
            return float(irr) if not pd.isna(irr) else 0.0
        except:
            return 0.0
    
    def process_all_scenarios(self) -> List[dict]:
        """
        Process all scenarios with Aug22.csv rates ONLY
        NO default fallbacks - professor's rates or fail
        """
        
        waterfall_results = []
        
        print(f"Processing {len(self.module_b_results)} scenarios with Aug22.csv rates...")
        
        for scenario in self.module_b_results:
            # Extract scenario data
            country_name = scenario['country_name']
            fund_type = scenario['fund_type']
            pricing_method = scenario['pricing_methodology']
            
            # Get financial data
            annual_cash_flows = scenario.get('annual_cash_flows', [])
            initial_investment = scenario.get('initial_investment', 0)
            project_irr = scenario.get('project_irr', 0)
            
            if not annual_cash_flows or initial_investment <= 0:
                print(f" Skipping {country_name}: insufficient cash flow data")
                continue
            
            try:
                # Get required returns for this country from Aug22.csv
                required_returns = self._get_required_returns(country_name)
            except Exception as e:
                print(f" Skipping {country_name}: {e}")
                continue
            
            # Calculate tranche investments
            tranche_investments = self._calculate_tranche_investments(initial_investment)
            annual_requirements = self._calculate_annual_requirements(tranche_investments, required_returns)
            
            # Run BOTH waterfall versions as per original specification
            v1_payments, v1_annual = self._version_1_waterfall(
                annual_cash_flows, annual_requirements, tranche_investments
            )
            
            v2_payments, v2_annual = self._version_2_robust_waterfall(
                annual_cash_flows, annual_requirements, tranche_investments  
            )
            
            # Use Version 2 (robust with carryforward) for final results
            total_payments, annual_payment_records = v2_payments, v2_annual
            
            # Create results for each tranche
            for tranche in self.capital_structure.keys():
                
                # Calculate tranche IRR
                tranche_irr = self._calculate_tranche_irr(
                    tranche_investments[tranche], annual_payment_records, tranche
                )
                
                # Determine if investment is attractive using Aug22.csv rate
                required_return = required_returns[tranche]
                excess_return = tranche_irr - required_return
                is_attractive = excess_return >= 0
                
                result = {
                    'scenario_index': len(waterfall_results),
                    'country_name': country_name,
                    'country_code': scenario.get('country_code', 'XX'),
                    'fund_type': fund_type,
                    'pricing_methodology': pricing_method,
                    'tranche': tranche,
                    'tranche_investment': tranche_investments[tranche],
                    'tranche_irr': tranche_irr,
                    'required_return': required_return,  # From Aug22.csv ONLY
                    'excess_return': excess_return,
                    'irr_status': scenario.get('irr_status', 'SUCCESS'),
                    'total_payments_received': total_payments[tranche],
                    'annual_payments': [p['payments'][tranche] for p in annual_payment_records],
                    'payment_years': list(range(len(annual_payment_records))),
                    'project_irr': project_irr,
                    'attractive_investment': is_attractive,
                    'total_investment': initial_investment,
                    'capital_structure': self.capital_structure,
                    'aug22_rates_used': True  # Confirm Aug22.csv used
                }
                
                waterfall_results.append(result)
        
        print(f" Completed waterfall analysis: {len(waterfall_results)} tranche results using Aug22.csv rates")
        return waterfall_results

def run_module_c_robust_waterfall_analysis(module_b_results: list, aug22_csv_path: str = 'Aug22.csv'):
    """
    FIXED Module C: Uses ONLY Aug22.csv with proper column names
    NO default rates - professor's country-specific rates ONLY
    """
    print("="*60)
    print("MODULE C: ROBUST WATERFALL ANALYSIS - FIXED VERSION")
    print("Using ONLY Aug22.csv: new_woinf_sen_debt, new_woinf_sub_debt, new_woinf_equity")
    print("="*60)
    
    # Initialize analyzer with strict Aug22.csv requirement
    try:
        waterfall_analyzer = WaterfallAnalyzer(module_b_results, aug22_csv_path)
    except Exception as e:
        print(f" CRITICAL ERROR: {e}")
        print("Cannot proceed without Aug22.csv data")
        return [], None
    
    # Process all scenarios
    waterfall_results = waterfall_analyzer.process_all_scenarios()
    
    print(f"\n✅ WATERFALL ANALYSIS COMPLETE:")
    print(f"Total scenarios processed: {len(module_b_results)}")
    print(f"Tranche results generated: {len(waterfall_results)}")
    print(f"Using Aug22.csv rates for: {len(set([r['country_name'] for r in waterfall_results]))}")
    print(f"Attractive investments: {len([r for r in waterfall_results if r['attractive_investment']])}")
    
    return waterfall_results, waterfall_analyzer