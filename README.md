# carbon_arbitrage_returns
Subsidiary of carbon_arbitrage
Version 12 Execution Order: v2_data_loaders.py, v2_dcf_analyzer.py, irr_calculator.py, waterfall_analyzer.py, optimization_engine.py, visualization_suite.py, renewable_energy_pipeline.py

SEP 11 UPDATES
v2_data_loaders.py vs Original data_loaders.py

  Key Fixes:
  • Column Naming: Uses Technology_Main (matches CSV) instead of incorrect primary_technology
  • Parameter Naming: Uses carbon_arbitrage_returns parameter (clearer hierarchy naming)
  • Missing Region Handling: Falls back to 'global' 

  Robust Extrapolation Logic:
  • Partial Country Extrapolation: Countries with some tech data use their own country average for missing
  technologies (prevents contamination)
  • Pure Regional Averages: Only uses direct data for regional calculations (no extrapolated-from-extrapolated data)

  Meeting Requirements:
  • Technology-Specific Pricing: Calculates (country, technology) combinations with capacity-weighted averages
  • Visualization Added: Country-technology scatter plot 

  Data Quality:
  • Cleaner Extrapolation: Direct → Country Average → Regional → Global hierarchy
  • Export Format: Uses Technology_Main column consistently for downstream DCF analysis


v2_dcf_analyzer.py vs. original dcf_analyzer.py 

  Overview
  Discounted Cash Flow (DCF) analysis for renewable energy investments across developing countries, implementing
  LaTeX mathematical specifications with three pricing scenarios for two fund types (commitment until 2035 vs 2050)

  Mathematical Model

  Core Formula:
  Net Cash Flow: CFi,t = Ri,t - Ii,t - OCi,t
  NPV = Σ(t=2025 to end_year) CFi,t / (1 + 0.02795381840850683)^(t-2025)

  Components:
  - Investment Cash Flow: Ii,t = Σ(installed_capacity_kW × unit_cost_$/kW)
  - Revenue Cash Flow: Ri,t = Σ((available_capacity_kW × CF × 8760/1000) × price_$/MWh)
  - Opportunity Cost: OCi,t = opportunity_cost_trillions × 1e12

  Fund Scenarios

  | Fund                   | Investment Period | Revenue Period | Description                                 |
  |------------------------|-------------------|----------------|---------------------------------------------|
  | Fund A (2035)          | 2025-2035         | 2025-2065      | 10-year commitment + 30-year asset lifetime |
  | Fund B (Net Zero 2050) | 2025-2050         | 2025-2080      | 25-year commitment + 30-year asset lifetime |

  Pricing Scenarios

  Extracts three pricing methodologies from v2_data_loaders output:
  - Weighted Average: Capacity-weighted mean pricing
  - Minimum: Conservative scenario using lowest observed prices
  - Maximum: Optimistic scenario using highest observed prices

  Input Files & Units

  Capacity Data
  - battery_yearly_installed_capacity_[2035|Net Zero 2050]_XX.json: Investment stream (kW)
  - battery_yearly_available_capacity_[2035|Net Zero 2050]_XX.json: Revenue stream (kW)
  - battery_unit_ic_XX.json: Unit costs ($/kW, excludes short/long battery variants)

  Supporting Data
  - v2_revenue_factor_table_YYYYMMDD.csv: Technology-specific pricing from v2_data_loaders ($/MWh)
  - v3_capacity_weighted_average_capacity_factor.csv: Capacity factors by technology (fraction)
  - yearly_by_country_opportunity_cost_non_discounted_main.csv: Phase-out costs (trillions USD)

  Unit Conversions

  Energy Conversion:
  kW × capacity_factor × 8760_hours/year ÷ 1000 = MWh/year
  MWh/year × price_$/MWh = revenue_$/year

  Cost Conversions:
  kW × unit_cost_$/kW = investment_$/year
  trillions_USD × 1e12 = opportunity_cost_$/year

  Output Structure

  Expected Results: Countries × 2 funds × 3 pricing = ~600 scenarios

  Files Generated

  1. v2_dcf_npv_summary_all_funds_YYYYMMDD_HHMM.csv: NPV analysis summary
  2. v2_dcf_cash_flows_all_funds_YYYYMMDD_HHMM.csv: Detailed yearly cash flows (undiscounted for IRR/waterfall
  modules)

  Key Columns

  - NPV Summary: country_iso2, fund_name, pricing_scenario, npv_usd, total_investment_usd, total_revenue_usd
  - Cash Flows: year, investment_cf, revenue_cf, opportunity_cost_cf, net_cf (undiscounted), discounted_net_cf

  Integration

  - Input: Requires v2_data_loaders output for revenue factors
  - Output: Provides undiscounted cash flows for irr_calculator and waterfall_analyzer modules
  - Discount Rate: LaTeX-specified value (0.02795381840850683) for consistent NPV calculations
