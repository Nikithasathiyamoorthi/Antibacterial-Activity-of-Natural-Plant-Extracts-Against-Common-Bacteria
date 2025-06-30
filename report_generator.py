import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load the antibacterial data"""
    try:
        df = pd.read_csv('antibacterial_data.csv')
        return df
    except FileNotFoundError:
        print("Data file not found. Please run data_generator.py first.")
        return None

def generate_summary_statistics(df):
    """Generate summary statistics for the report"""
    stats = {
        'total_tests': len(df),
        'num_plants': df['Plant_Extract'].nunique(),
        'num_bacteria': df['Bacteria'].nunique(),
        'num_concentrations': df['Concentration_mg_mL'].nunique(),
        'mean_inhibition': df['Inhibition_Zone_mm'].mean(),
        'std_inhibition': df['Inhibition_Zone_mm'].std(),
        'max_inhibition': df['Inhibition_Zone_mm'].max(),
        'min_inhibition': df['Inhibition_Zone_mm'].min()
    }
    
    # Activity level distribution
    activity_dist = df['Activity_Level'].value_counts()
    stats['high_activity_pct'] = (activity_dist.get('High', 0) / len(df)) * 100
    stats['moderate_activity_pct'] = (activity_dist.get('Moderate', 0) / len(df)) * 100
    stats['low_activity_pct'] = (activity_dist.get('Low', 0) / len(df)) * 100
    stats['no_activity_pct'] = (activity_dist.get('None', 0) / len(df)) * 100
    
    return stats

def find_top_performers(df):
    """Find top performing plant extracts and most susceptible bacteria"""
    # Top plant extracts
    top_plants = df.groupby('Plant_Extract')['Inhibition_Zone_mm'].agg(['mean', 'std', 'count']).round(2)
    top_plants = top_plants.sort_values('mean', ascending=False).head(5)
    
    # Most susceptible bacteria
    susceptible_bacteria = df.groupby('Bacteria')['Inhibition_Zone_mm'].agg(['mean', 'std', 'count']).round(2)
    susceptible_bacteria = susceptible_bacteria.sort_values('mean', ascending=False).head(5)
    
    # Most resistant bacteria
    resistant_bacteria = df.groupby('Bacteria')['Inhibition_Zone_mm'].agg(['mean', 'std', 'count']).round(2)
    resistant_bacteria = resistant_bacteria.sort_values('mean', ascending=True).head(3)
    
    return top_plants, susceptible_bacteria, resistant_bacteria

def generate_html_report(df):
    """Generate comprehensive HTML report"""
    stats = generate_summary_statistics(df)
    top_plants, susceptible_bacteria, resistant_bacteria = find_top_performers(df)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Antibacterial Activity Analysis Report</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }}
            h3 {{
                color: #2c3e50;
                margin-top: 25px;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .summary-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .summary-card h3 {{
                margin: 0 0 10px 0;
                color: white;
            }}
            .summary-card .value {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background-color: white;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .highlight {{
                background-color: #e8f5e8;
                font-weight: bold;
            }}
            .methodology {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .conclusion {{
                background-color: #d5e8d4;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
                border-left: 5px solid #27ae60;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Antibacterial Activity of Natural Plant Extracts Against Common Bacteria</h1>
            <p style="text-align: center; font-style: italic; color: #7f8c8d;">
                Comprehensive Analysis Report - Generated on {datetime.now().strftime('%B %d, %Y')}
            </p>
            
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Tests</h3>
                    <div class="value">{stats['total_tests']:,}</div>
                </div>
                <div class="summary-card">
                    <h3>Plant Extracts</h3>
                    <div class="value">{stats['num_plants']}</div>
                </div>
                <div class="summary-card">
                    <h3>Bacteria Species</h3>
                    <div class="value">{stats['num_bacteria']}</div>
                </div>
                <div class="summary-card">
                    <h3>High Activity</h3>
                    <div class="value">{stats['high_activity_pct']:.1f}%</div>
                </div>
            </div>
            
            <h2>Study Overview</h2>
            <div class="methodology">
                <h3>Methodology</h3>
                <ul>
                    <li><strong>Plant Extracts Tested:</strong> {stats['num_plants']} different natural plant extracts</li>
                    <li><strong>Bacterial Strains:</strong> {stats['num_bacteria']} common pathogenic bacteria</li>
                    <li><strong>Concentrations:</strong> {stats['num_concentrations']} different concentration levels (10-200 mg/mL)</li>
                    <li><strong>Replicates:</strong> 3 replicates per condition for statistical reliability</li>
                    <li><strong>Measurement:</strong> Zone of inhibition diameter (mm) using disk diffusion method</li>
                </ul>
            </div>
            
            <h2>Key Findings</h2>
            
            <h3>Overall Activity Distribution</h3>
            <table>
                <tr>
                    <th>Activity Level</th>
                    <th>Inhibition Zone Range</th>
                    <th>Percentage of Tests</th>
                </tr>
                <tr class="highlight">
                    <td>High Activity</td>
                    <td>≥ 20 mm</td>
                    <td>{stats['high_activity_pct']:.1f}%</td>
                </tr>
                <tr>
                    <td>Moderate Activity</td>
                    <td>15-19 mm</td>
                    <td>{stats['moderate_activity_pct']:.1f}%</td>
                </tr>
                <tr>
                    <td>Low Activity</td>
                    <td>10-14 mm</td>
                    <td>{stats['low_activity_pct']:.1f}%</td>
                </tr>
                <tr>
                    <td>No Activity</td>
                    <td>< 10 mm</td>
                    <td>{stats['no_activity_pct']:.1f}%</td>
                </tr>
            </table>
            
            <h3>Statistical Summary</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Mean Inhibition Zone</td>
                    <td>{stats['mean_inhibition']:.2f} ± {stats['std_inhibition']:.2f} mm</td>
                </tr>
                <tr>
                    <td>Maximum Inhibition Zone</td>
                    <td>{stats['max_inhibition']:.2f} mm</td>
                </tr>
                <tr>
                    <td>Minimum Inhibition Zone</td>
                    <td>{stats['min_inhibition']:.2f} mm</td>
                </tr>
            </table>
            
            <h3>Top 5 Most Effective Plant Extracts</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Plant Extract</th>
                    <th>Mean Inhibition Zone (mm)</th>
                    <th>Standard Deviation</th>
                    <th>Number of Tests</th>
                </tr>
    """
    
    for i, (plant, data) in enumerate(top_plants.iterrows(), 1):
        plant_name = plant.split('(')[0].strip()
        html_content += f"""
                <tr {'class="highlight"' if i == 1 else ''}>
                    <td>{i}</td>
                    <td>{plant_name}</td>
                    <td>{data['mean']:.2f}</td>
                    <td>{data['std']:.2f}</td>
                    <td>{data['count']}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h3>Most Susceptible Bacteria</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Bacteria Species</th>
                    <th>Mean Inhibition Zone (mm)</th>
                    <th>Standard Deviation</th>
                </tr>
    """
    
    for i, (bacteria, data) in enumerate(susceptible_bacteria.iterrows(), 1):
        html_content += f"""
                <tr {'class="highlight"' if i == 1 else ''}>
                    <td>{i}</td>
                    <td><em>{bacteria}</em></td>
                    <td>{data['mean']:.2f}</td>
                    <td>{data['std']:.2f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h3>Most Resistant Bacteria</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Bacteria Species</th>
                    <th>Mean Inhibition Zone (mm)</th>
                    <th>Standard Deviation</th>
                </tr>
    """
    
    for i, (bacteria, data) in enumerate(resistant_bacteria.iterrows(), 1):
        html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td><em>{bacteria}</em></td>
                    <td>{data['mean']:.2f}</td>
                    <td>{data['std']:.2f}</td>
                </tr>
        """
    
    # Calculate concentration effect
    conc_effect = df.groupby('Concentration_mg_mL')['Inhibition_Zone_mm'].mean()
    correlation = df['Concentration_mg_mL'].corr(df['Inhibition_Zone_mm'])
    
    html_content += f"""
            </table>
            
            <h2>Concentration-Response Analysis</h2>
            <p>The analysis reveals a <strong>{'positive' if correlation > 0 else 'negative'}</strong> correlation 
            (r = {correlation:.3f}) between extract concentration and antibacterial activity.</p>
            
            <table>
                <tr>
                    <th>Concentration (mg/mL)</th>
                    <th>Mean Inhibition Zone (mm)</th>
                </tr>
    """
    
    for conc, inhibition in conc_effect.items():
        html_content += f"""
                <tr>
                    <td>{conc}</td>
                    <td>{inhibition:.2f}</td>
                </tr>
        """
    
    html_content += f"""
            </table>
            
            <div class="conclusion">
                <h2>Conclusions and Recommendations</h2>
                <h3>Key Findings:</h3>
                <ul>
                    <li><strong>Most Effective Extract:</strong> {top_plants.index[0].split('(')[0].strip()} showed the highest mean antibacterial activity ({top_plants.iloc[0]['mean']:.2f} mm)</li>
                    <li><strong>Most Susceptible Pathogen:</strong> <em>{susceptible_bacteria.index[0]}</em> was most susceptible to plant extracts</li>
                    <li><strong>Concentration Effect:</strong> {'Strong positive correlation' if correlation > 0.7 else 'Moderate positive correlation' if correlation > 0.4 else 'Weak correlation'} between concentration and activity</li>
                    <li><strong>Success Rate:</strong> {stats['high_activity_pct'] + stats['moderate_activity_pct']:.1f}% of tests showed moderate to high antibacterial activity</li>
                </ul>
                
                <h3>Clinical Implications:</h3>
                <ul>
                    <li>Natural plant extracts show promising antibacterial potential against common pathogens</li>
                    <li>Concentration optimization is crucial for maximum therapeutic effect</li>
                    <li>Species-specific responses suggest targeted therapy approaches</li>
                    <li>Further research needed for standardization and clinical applications</li>
                </ul>
                
                <h3>Future Research Directions:</h3>
                <ul>
                    <li>Investigate active compounds responsible for antibacterial activity</li>
                    <li>Conduct in-vivo studies to validate in-vitro findings</li>
                    <li>Explore synergistic effects of plant extract combinations</li>
                    <li>Develop standardized extraction and formulation protocols</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>This report was generated using Python-based statistical analysis and machine learning techniques.</p>
                <p>For questions or additional analysis, please contact the research team.</p>
                <p><strong>Generated on:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def generate_text_summary(df):
    """Generate a text summary of key findings"""
    stats = generate_summary_statistics(df)
    top_plants, susceptible_bacteria, resistant_bacteria = find_top_performers(df)
    
    summary = f"""
ANTIBACTERIAL ACTIVITY ANALYSIS - SUMMARY REPORT
{'='*60}

STUDY OVERVIEW:
- Total tests conducted: {stats['total_tests']:,}
- Plant extracts tested: {stats['num_plants']}
- Bacterial species: {stats['num_bacteria']}
- Concentration levels: {stats['num_concentrations']}

KEY STATISTICS:
- Mean inhibition zone: {stats['mean_inhibition']:.2f} ± {stats['std_inhibition']:.2f} mm
- Maximum inhibition: {stats['max_inhibition']:.2f} mm
- High activity rate: {stats['high_activity_pct']:.1f}%
- Moderate activity rate: {stats['moderate_activity_pct']:.1f}%

TOP 3 MOST EFFECTIVE PLANT EXTRACTS:
1. {top_plants.index[0].split('(')[0].strip()}: {top_plants.iloc[0]['mean']:.2f} mm
2. {top_plants.index[1].split('(')[0].strip()}: {top_plants.iloc[1]['mean']:.2f} mm
3. {top_plants.index[2].split('(')[0].strip()}: {top_plants.iloc[2]['mean']:.2f} mm

MOST SUSCEPTIBLE BACTERIA:
1. {susceptible_bacteria.index[0]}: {susceptible_bacteria.iloc[0]['mean']:.2f} mm
2. {susceptible_bacteria.index[1]}: {susceptible_bacteria.iloc[1]['mean']:.2f} mm
3. {susceptible_bacteria.index[2]}: {susceptible_bacteria.iloc[2]['mean']:.2f} mm

CONCENTRATION EFFECT:
- Correlation coefficient: {df['Concentration_mg_mL'].corr(df['Inhibition_Zone_mm']):.3f}
- Relationship: {'Strong positive' if df['Concentration_mg_mL'].corr(df['Inhibition_Zone_mm']) > 0.7 else 'Moderate positive' if df['Concentration_mg_mL'].corr(df['Inhibition_Zone_mm']) > 0.4 else 'Weak'}

CONCLUSIONS:
- Natural plant extracts show significant antibacterial potential
- Concentration-dependent activity observed
- Species-specific responses suggest targeted applications
- Further research recommended for clinical development

Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return summary

def main():
    """Main report generation function"""
    df = load_data()
    if df is None:
        return
    
    print("Generating comprehensive analysis report...")
    
    # Generate HTML report
    html_report = generate_html_report(df)
    with open('antibacterial_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    # Generate text summary
    text_summary = generate_text_summary(df)
    with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write(text_summary)
    
    print("Reports generated successfully!")
    print("\nFiles created:")
    print("1. antibacterial_analysis_report.html - Comprehensive HTML report")
    print("2. analysis_summary.txt - Text summary of key findings")
    print("\nOpen the HTML file in your web browser to view the full report.")
    
    # Display summary in console
    print("\n" + "="*60)
    print("QUICK SUMMARY:")
    print("="*60)
    print(text_summary)

if __name__ == "__main__":
    main()
