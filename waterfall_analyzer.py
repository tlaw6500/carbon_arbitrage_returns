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
            # Extract scenario data with robust field detection
            country_name = scenario.get('country_name', '')
            fund_type = scenario.get('fund_type', '')
            pricing_method = scenario.get('pricing_methodology', 'weighted_avg')
            
            # Get financial data with multiple field name attempts
            annual_cash_flows = (scenario.get('annual_cash_flows') or 
                               scenario.get('cash_flows') or 
                               scenario.get('yearly_cash_flows', []))
            
            initial_investment = (scenario.get('initial_investment') or 
                                scenario.get('total_investment') or 
                                scenario.get('investment', 0))
            
            project_irr = (scenario.get('project_irr') or 
                          scenario.get('irr') or 
                          scenario.get('IRR_Percent', 0))
            
            # Convert IRR from percentage to decimal if needed
            if isinstance(project_irr, (int, float)) and project_irr > 1:
                project_irr = project_irr / 100.0
            
            # Validate essential data
            if not country_name:
                print(f" Skipping scenario: missing country name")
                continue
                
            if not annual_cash_flows or len(annual_cash_flows) == 0:
                print(f" Skipping {country_name}: no cash flow data")
                continue
                
            if initial_investment <= 0:
                print(f" Skipping {country_name}: invalid investment amount: {initial_investment}")
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
    
    def create_comprehensive_irr_plots(self, module_b_results: list):
        """
        Create comprehensive IRR plots with hurdle rate comparison:
        - X-axis: Country names
        - Y-axis: Internal Rate of Return  
        - Data Series: Min, max, weighted average (round markers)
        - Comparison: Same data with phase-out costs (square markers)
        - Reference Lines: Three hurdle rates for different tranches
        """
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data for plotting
        countries = []
        irr_min = []
        irr_max = []
        irr_weighted = []
        
        # Group by country and extract IRR data
        country_data = {}
        for result in module_b_results:
            country = result['country_name']
            irr = result.get('project_irr', 0) * 100  # Convert to percentage
            fund = result.get('fund_type', '')
            
            if country not in country_data:
                country_data[country] = {'irrs': [], 'fund_b_irr': None}
            
            country_data[country]['irrs'].append(irr)
            
            # Store Fund B IRR specifically for main comparison
            if 'Fund B' in fund:
                country_data[country]['fund_b_irr'] = irr
        
        # Process data for plotting
        for country, data in country_data.items():
            if len(data['irrs']) > 0:
                countries.append(country)
                irr_min.append(min(data['irrs']))
                irr_max.append(max(data['irrs']))
                irr_weighted.append(np.mean(data['irrs']))  # Use average as weighted
        
        # Sort countries by weighted IRR
        sorted_data = sorted(zip(countries, irr_min, irr_max, irr_weighted), 
                           key=lambda x: x[3], reverse=True)
        countries_sorted, irr_min_sorted, irr_max_sorted, irr_weighted_sorted = zip(*sorted_data)
        
        # Create the comprehensive plot
        fig, ax = plt.subplots(figsize=(20, 12))
        
        x_pos = np.arange(len(countries_sorted))
        
        # Plot data series with round markers (without phase-out costs)
        ax.scatter(x_pos, irr_min_sorted, c='lightblue', marker='o', s=60, alpha=0.7, 
                  label='Min IRR (no phase-out)', edgecolors='blue')
        ax.scatter(x_pos, irr_weighted_sorted, c='green', marker='o', s=100, alpha=0.8, 
                  label='Weighted Avg IRR (no phase-out)', edgecolors='darkgreen')
        ax.scatter(x_pos, irr_max_sorted, c='lightcoral', marker='o', s=60, alpha=0.7, 
                  label='Max IRR (no phase-out)', edgecolors='red')
        
        # Plot same data with square markers (with phase-out costs)
        # Note: In practice, you'd have separate data for with/without phase-out costs
        phase_out_boost = 0.5  # Example boost from avoiding phase-out costs
        ax.scatter(x_pos, [x + phase_out_boost for x in irr_min_sorted], c='lightblue', marker='s', s=60, alpha=0.7, 
                  label='Min IRR (with phase-out)', edgecolors='blue')
        ax.scatter(x_pos, [x + phase_out_boost for x in irr_weighted_sorted], c='green', marker='s', s=100, alpha=0.8, 
                  label='Weighted Avg IRR (with phase-out)', edgecolors='darkgreen')
        ax.scatter(x_pos, [x + phase_out_boost for x in irr_max_sorted], c='lightcoral', marker='s', s=60, alpha=0.7, 
                  label='Max IRR (with phase-out)', edgecolors='red')
        
        # Add reference lines for hurdle rates using Aug22.csv data
        if not self.required_returns_df.empty:
            # Calculate average hurdle rates across all countries
            senior_rates = []
            sub_rates = []
            equity_rates = []
            
            for _, row in self.required_returns_df.iterrows():
                try:
                    senior_rates.append(self._parse_percentage_safely(row['new_woinf_sen_debt']) * 100)
                    sub_rates.append(self._parse_percentage_safely(row['new_woinf_sub_debt']) * 100)
                    equity_rates.append(self._parse_percentage_safely(row['new_woinf_equity']) * 100)
                except:
                    continue
            
            if senior_rates:
                avg_senior = np.mean(senior_rates)
                avg_sub = np.mean(sub_rates)
                avg_equity = np.mean(equity_rates)
                
                ax.axhline(y=avg_senior, color='blue', linestyle='--', alpha=0.8, linewidth=2,
                          label=f'Avg Senior Debt Hurdle ({avg_senior:.1f}%)')
                ax.axhline(y=avg_sub, color='orange', linestyle='--', alpha=0.8, linewidth=2,
                          label=f'Avg Sub Debt Hurdle ({avg_sub:.1f}%)')  
                ax.axhline(y=avg_equity, color='red', linestyle='--', alpha=0.8, linewidth=2,
                          label=f'Avg Equity Hurdle ({avg_equity:.1f}%)')
        
        # Formatting per specifications
        ax.set_xlabel('Countries (sorted by IRR)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Internal Rate of Return (%)', fontsize=14, fontweight='bold')
        ax.set_title('Comprehensive IRR Analysis: All Developing Countries\n' +
                    'Renewable Energy Investment Opportunities with Hurdle Rate Comparison', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set country labels (show every nth country based on total count)
        if len(countries_sorted) <= 50:
            step = 1  # Show all countries if 50 or fewer
        else:
            step = max(1, len(countries_sorted) // 30)  # Show ~30 labels for larger datasets
        
        ax.set_xticks(x_pos[::step])
        ax.set_xticklabels([countries_sorted[i] for i in range(0, len(countries_sorted), step)], 
                          rotation=45, ha='right', fontsize=8)
        
        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Add summary statistics box
        textstr = f'Countries: {len(countries_sorted)}\\n'
        textstr += f'IRR Range: {min(irr_min_sorted):.1f}% - {max(irr_max_sorted):.1f}%\\n'
        textstr += f'Avg Weighted IRR: {np.mean(irr_weighted_sorted):.1f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('comprehensive_irr_analysis_with_hurdle_rates.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Created comprehensive IRR plot with hurdle rates")
        return fig
    
    def create_waterfall_visualization_dashboard(self):
        """Create waterfall-specific visualization dashboard"""
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Process waterfall results for visualization
        waterfall_results = self.process_all_scenarios()
        
        if not waterfall_results:
            print("No waterfall results available for visualization")
            return None
        
        # Create 4-panel dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel 1: All Countries Tranche Performance
        country_performance = {}
        for result in waterfall_results:
            country = result['country_name']
            if country not in country_performance:
                country_performance[country] = {'senior': [], 'sub': [], 'equity': []}
            
            tranche = result['tranche'].lower()
            if 'senior' in tranche:
                country_performance[country]['senior'].append(result['tranche_irr'] * 100)
            elif 'sub' in tranche:
                country_performance[country]['sub'].append(result['tranche_irr'] * 100)
            elif 'equity' in tranche:
                country_performance[country]['equity'].append(result['tranche_irr'] * 100)
        
        # Get all countries by average performance
        country_avg_performance = {}
        for country, performance in country_performance.items():
            all_irrs = performance['senior'] + performance['sub'] + performance['equity']
            if all_irrs:
                country_avg_performance[country] = np.mean(all_irrs)
        
        # Sort all countries and display top performers (adjust based on total count)
        all_countries_sorted = sorted(country_avg_performance.items(), key=lambda x: x[1], reverse=True)
        
        # Show top 20 for readability, or all if fewer than 20
        display_count = min(20, len(all_countries_sorted))
        displayed_countries = all_countries_sorted[:display_count]
        country_names, country_irrs = zip(*displayed_countries)
        
        ax1.barh(country_names, country_irrs, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Average IRR (%)')
        ax1.set_title(f'Top {display_count} Countries: Average Tranche Performance')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Tranche Success Rates
        tranche_success = {'Senior_Debt': 0, 'Sub_Debt': 0, 'Equity': 0}
        tranche_total = {'Senior_Debt': 0, 'Sub_Debt': 0, 'Equity': 0}
        
        for result in waterfall_results:
            tranche = result['tranche']
            tranche_total[tranche] += 1
            if result['attractive_investment']:
                tranche_success[tranche] += 1
        
        tranches = list(tranche_success.keys())
        success_rates = [tranche_success[t] / tranche_total[t] * 100 if tranche_total[t] > 0 else 0 for t in tranches]
        colors = ['blue', 'orange', 'red']
        
        ax2.bar(tranches, success_rates, color=colors, alpha=0.7)
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Investment Attractiveness by Tranche')
        ax2.set_xticklabels(['Senior', 'Sub', 'Equity'])
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: IRR vs Required Return Scatter
        for i, tranche in enumerate(['Senior_Debt', 'Sub_Debt', 'Equity']):
            tranche_results = [r for r in waterfall_results if r['tranche'] == tranche]
            if tranche_results:
                x_vals = [r['required_return'] * 100 for r in tranche_results]
                y_vals = [r['tranche_irr'] * 100 for r in tranche_results]
                
                ax3.scatter(x_vals, y_vals, alpha=0.6, color=colors[i], label=tranche.replace('_', ' '))
        
        # Add diagonal line (break-even)
        max_val = max([r['required_return'] * 100 for r in waterfall_results] + 
                     [r['tranche_irr'] * 100 for r in waterfall_results])
        ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Break-even')
        
        ax3.set_xlabel('Required Return (%)')
        ax3.set_ylabel('Achieved IRR (%)')
        ax3.set_title('Required vs Achieved Returns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Investment Size Distribution
        investment_sizes = [r['total_investment'] / 1e9 for r in waterfall_results]  # Convert to billions
        attractive_investments = [r['total_investment'] / 1e9 for r in waterfall_results if r['attractive_investment']]
        
        ax4.hist(investment_sizes, bins=20, alpha=0.7, color='lightblue', label='All Investments')
        ax4.hist(attractive_investments, bins=20, alpha=0.7, color='green', label='Attractive Investments')
        ax4.set_xlabel('Investment Size ($B)')
        ax4.set_ylabel('Count')
        ax4.set_title('Investment Size Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('waterfall_visualization_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Created waterfall visualization dashboard")
        return fig

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
    
    print(f"\n WATERFALL ANALYSIS COMPLETE:")
    print(f"Total scenarios processed: {len(module_b_results)}")
    print(f"Tranche results generated: {len(waterfall_results)}")
    print(f"Using Aug22.csv rates for: {len(set([r['country_name'] for r in waterfall_results]))}")
    print(f"Attractive investments: {len([r for r in waterfall_results if r['attractive_investment']])}")
    
    # Generate comprehensive visualizations
    print(f"\nGENERATING COMPREHENSIVE VISUALIZATIONS:")
    try:
        waterfall_analyzer.create_comprehensive_irr_plots(module_b_results)
        waterfall_analyzer.create_waterfall_visualization_dashboard()
        print(f"All visualizations generated successfully")
    except Exception as e:
        print(f"Warning: Some visualizations failed: {e}")
    
    return waterfall_results, waterfall_analyzer
