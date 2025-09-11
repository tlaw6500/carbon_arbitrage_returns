"""
V2 Data Loaders - Technology-Specific Revenue Analysis
AK Decarbonise Project
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class V2EnhancedRevenueAnalyzer:
    
    def __init__(self, csv_path='renewable_energy_revenue_table_restructured_v10.csv',
                 carbon_arbitrage_returns='./carbon_arbitrage_code'):
        
        self.csv_path = csv_path
        self.carbon_arbitrage_returns = carbon_arbitrage_returns
        
        self.load_data()
        self.calculate_direct_pricing()
        self.extrapolate_missing_data()
        self.create_visualization()
        self.export_revenue_table()
    
    def load_data(self):
        self.df = pd.read_csv(self.csv_path, encoding='latin-1')
        self.df['Revenue/MWh (USD 2023)'] = pd.to_numeric(self.df['Revenue/MWh (USD 2023)'], errors='coerce')
        self.df['Capacity (MW)'] = pd.to_numeric(self.df['Capacity (MW)'], errors='coerce')
        self.df = self.df.dropna(subset=['Revenue/MWh (USD 2023)', 'Capacity (MW)'])
        
        # Load regional mapping
        region_path = os.path.join(self.carbon_arbitrage_returns, 'data', 'country_ISO-3166_with_region.csv')
        regions = pd.read_csv(region_path)
        self.region_map = dict(zip(regions['alpha-2'], regions['region-code']))
        
        self.df_dev = self.df[self.df['UNFCCC_Classification'] == 'Developing'].copy()
        self.technologies = sorted(self.df_dev['Technology_Main'].unique())
        
    def calculate_direct_pricing(self):
        results = []
        
        for country in self.df_dev['Country_ISO2'].unique():
            country_data = self.df_dev[self.df_dev['Country_ISO2'] == country]
            region_code = self.region_map.get(country, 'global')
            
            # Calculate country average first for partial extrapolation
            country_avg = self.get_country_weighted_average(country_data)
            
            for technology in self.technologies:
                tech_data = country_data[country_data['Technology_Main'] == technology]
                
                if len(tech_data) > 0:
                    # Direct data
                    total_capacity = tech_data['Capacity (MW)'].sum()
                    
                    if total_capacity > 0:
                        weights = tech_data['Capacity (MW)'] / total_capacity
                        weighted_avg = (weights * tech_data['Revenue/MWh (USD 2023)']).sum()
                    else:
                        weighted_avg = tech_data['Revenue/MWh (USD 2023)'].mean()
                    
                    results.append({
                        'country_iso2': country,
                        'Technology_Main': technology,
                        'region_code': region_code,
                        'weighted_avg_usd_mwh': round(weighted_avg, 3),
                        'min_usd_mwh': round(tech_data['Revenue/MWh (USD 2023)'].min(), 3),
                        'max_usd_mwh': round(tech_data['Revenue/MWh (USD 2023)'].max(), 3),
                        'total_capacity_mw': round(total_capacity, 3),
                        'deal_count': len(tech_data),
                        'data_source': 'direct'
                    })
                else:
                    # Use country average for missing tech (partial extrapolation)
                    results.append({
                        'country_iso2': country,
                        'Technology_Main': technology,
                        'region_code': region_code,
                        'weighted_avg_usd_mwh': round(country_avg['weighted_avg'], 3),
                        'min_usd_mwh': round(country_avg['min'], 3),
                        'max_usd_mwh': round(country_avg['max'], 3),
                        'total_capacity_mw': 0.0,
                        'deal_count': 0,
                        'data_source': 'country_avg'
                    })
        
        self.direct_pricing = pd.DataFrame(results)
        
    def get_country_weighted_average(self, country_data):
        total_capacity = country_data['Capacity (MW)'].sum()
        if total_capacity > 0:
            weights = country_data['Capacity (MW)'] / total_capacity
            weighted_avg = (weights * country_data['Revenue/MWh (USD 2023)']).sum()
        else:
            weighted_avg = country_data['Revenue/MWh (USD 2023)'].mean()
        
        return {
            'weighted_avg': weighted_avg,
            'min': country_data['Revenue/MWh (USD 2023)'].min(),
            'max': country_data['Revenue/MWh (USD 2023)'].max()
        }
    
    def extrapolate_missing_data(self):
        # Only use direct data for regional averages
        direct_only = self.direct_pricing[self.direct_pricing['data_source'] == 'direct'].copy()
        
        # Regional averages by technology (only from direct data)
        regional_avg = {}
        for region_code in direct_only['region_code'].unique():
            region_data = direct_only[direct_only['region_code'] == region_code]
            
            for technology in self.technologies:
                tech_data = region_data[region_data['Technology_Main'] == technology]
                if len(tech_data) > 0:
                    total_cap = tech_data['total_capacity_mw'].sum()
                    if total_cap > 0:
                        weights = tech_data['total_capacity_mw'] / total_cap
                        weighted_avg = (weights * tech_data['weighted_avg_usd_mwh']).sum()
                    else:
                        weighted_avg = tech_data['weighted_avg_usd_mwh'].mean()
                    
                    regional_avg[(region_code, technology)] = {
                        'weighted_avg_usd_mwh': round(weighted_avg, 3),
                        'min_usd_mwh': round(tech_data['min_usd_mwh'].min(), 3),
                        'max_usd_mwh': round(tech_data['max_usd_mwh'].max(), 3)
                    }
        
        # Global averages (only from direct data)
        global_avg = {}
        for technology in self.technologies:
            tech_data = direct_only[direct_only['Technology_Main'] == technology]
            total_cap = tech_data['total_capacity_mw'].sum()
            if total_cap > 0:
                weights = tech_data['total_capacity_mw'] / total_cap
                weighted_avg = (weights * tech_data['weighted_avg_usd_mwh']).sum()
            else:
                weighted_avg = tech_data['weighted_avg_usd_mwh'].mean()
            
            global_avg[technology] = {
                'weighted_avg_usd_mwh': round(weighted_avg, 3),
                'min_usd_mwh': round(tech_data['min_usd_mwh'].min(), 3),
                'max_usd_mwh': round(tech_data['max_usd_mwh'].max(), 3)
            }
        
        # Add missing countries
        all_results = self.direct_pricing.copy()
        all_countries = set(self.region_map.keys())
        existing_combos = set(zip(self.direct_pricing['country_iso2'], self.direct_pricing['Technology_Main']))
        
        for country in all_countries:
            region_code = self.region_map.get(country, 'global')
            for technology in self.technologies:
                if (country, technology) not in existing_combos:
                    if (region_code, technology) in regional_avg:
                        data = regional_avg[(region_code, technology)]
                        source = 'regional'
                    else:
                        data = global_avg[technology]
                        source = 'global'
                    
                    all_results = pd.concat([all_results, pd.DataFrame([{
                        'country_iso2': country,
                        'Technology_Main': technology,
                        'region_code': region_code,
                        'weighted_avg_usd_mwh': data['weighted_avg_usd_mwh'],
                        'min_usd_mwh': data['min_usd_mwh'],
                        'max_usd_mwh': data['max_usd_mwh'],
                        'total_capacity_mw': 0.0,
                        'deal_count': 0,
                        'data_source': source
                    }])], ignore_index=True)
        
        self.final_table = all_results.sort_values(['country_iso2', 'Technology_Main'])
    
    def create_visualization(self):
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use raw deal data (not aggregated averages)
        countries = sorted(self.df_dev['Country_ISO2'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.technologies)))
        
        for i, technology in enumerate(self.technologies):
            tech_data = self.df_dev[self.df_dev['Technology_Main'] == technology]
            if len(tech_data) > 0:
                x_positions = [countries.index(country) for country in tech_data['Country_ISO2'] if country in countries]
                y_values = tech_data[tech_data['Country_ISO2'].isin(countries)]['Revenue/MWh (USD 2023)'].values[:len(x_positions)]
                
                ax.scatter(x_positions, y_values, color=colors[i], label=technology, alpha=0.7, s=60)
        
        ax.set_title('Deal Pricing by Country and Technology', fontweight='bold')
        ax.set_ylabel('Revenue (USD/MWh)')
        ax.set_xlabel('Countries')
        ax.set_xticks(range(len(countries)))
        ax.set_xticklabels(countries, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.4)
        
        plt.tight_layout()
        plt.savefig('v2_country_technology_pricing.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_revenue_table(self):
        output_file = f'v2_revenue_factor_table_{pd.Timestamp.now().strftime("%Y%m%d")}.csv'
        self.final_table.to_csv(output_file, index=False)
        return output_file
    
    def get_pricing(self, country_iso2, technology):
        result = self.final_table[
            (self.final_table['country_iso2'] == country_iso2) & 
            (self.final_table['Technology_Main'] == technology)
        ]
        return result.iloc[0].to_dict() if len(result) > 0 else None

def run_v2_analysis():
    analyzer = V2EnhancedRevenueAnalyzer()
    return analyzer

if __name__ == "__main__":
    analyzer = run_v2_analysis()
    print(f"Analysis complete: {len(analyzer.final_table)} entries exported")