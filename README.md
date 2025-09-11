# carbon_arbitrage_returns
Subsidiary of carbon_arbitrage
Version 12 Execution Order: v2_data_loaders.py, dcf_analyzer.py, irr_calculator.py, waterfall_analyzer.py, optimization_engine.py, visualization_suite.py, renewable_energy_pipeline.py

Sep 11 Updates
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
