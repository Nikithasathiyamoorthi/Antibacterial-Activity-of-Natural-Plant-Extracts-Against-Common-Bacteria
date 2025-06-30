import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, tukey_hsd
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the antibacterial data"""
    try:
        df = pd.read_csv('antibacterial_data.csv')
        return df
    except FileNotFoundError:
        print("Data file not found. Please run data_generator.py first.")
        return None

def descriptive_statistics(df):
    """Calculate descriptive statistics"""
    print("=== DESCRIPTIVE STATISTICS ===\n")
    
    # Overall statistics
    print("Overall Inhibition Zone Statistics:")
    print(df['Inhibition_Zone_mm'].describe())
    print()
    
    # Statistics by plant extract
    print("Statistics by Plant Extract:")
    plant_stats = df.groupby('Plant_Extract')['Inhibition_Zone_mm'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    print(plant_stats)
    print()
    
    # Statistics by bacteria
    print("Statistics by Bacteria:")
    bacteria_stats = df.groupby('Bacteria')['Inhibition_Zone_mm'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    print(bacteria_stats)
    print()
    
    # Activity level distribution
    print("Activity Level Distribution:")
    activity_dist = df['Activity_Level'].value_counts()
    activity_pct = df['Activity_Level'].value_counts(normalize=True) * 100
    
    for level in activity_dist.index:
        print(f"{level}: {activity_dist[level]} ({activity_pct[level]:.1f}%)")
    
    return plant_stats, bacteria_stats

def anova_analysis(df):
    """Perform ANOVA analysis"""
    print("\n=== ANOVA ANALYSIS ===\n")
    
    # ANOVA for plant extracts
    print("1. One-way ANOVA: Plant Extracts vs Inhibition Zone")
    plant_groups = [group['Inhibition_Zone_mm'].values for name, group in df.groupby('Plant_Extract')]
    f_stat_plants, p_val_plants = f_oneway(*plant_groups)
    
    print(f"F-statistic: {f_stat_plants:.4f}")
    print(f"P-value: {p_val_plants:.4e}")
    
    if p_val_plants < 0.05:
        print("Result: Significant difference between plant extracts (p < 0.05)")
    else:
        print("Result: No significant difference between plant extracts (p >= 0.05)")
    
    # ANOVA for bacteria
    print("\n2. One-way ANOVA: Bacteria vs Inhibition Zone")
    bacteria_groups = [group['Inhibition_Zone_mm'].values for name, group in df.groupby('Bacteria')]
    f_stat_bacteria, p_val_bacteria = f_oneway(*bacteria_groups)
    
    print(f"F-statistic: {f_stat_bacteria:.4f}")
    print(f"P-value: {p_val_bacteria:.4e}")
    
    if p_val_bacteria < 0.05:
        print("Result: Significant difference between bacteria (p < 0.05)")
    else:
        print("Result: No significant difference between bacteria (p >= 0.05)")
    
    # ANOVA for concentration
    print("\n3. One-way ANOVA: Concentration vs Inhibition Zone")
    conc_groups = [group['Inhibition_Zone_mm'].values for name, group in df.groupby('Concentration_mg_mL')]
    f_stat_conc, p_val_conc = f_oneway(*conc_groups)
    
    print(f"F-statistic: {f_stat_conc:.4f}")
    print(f"P-value: {p_val_conc:.4e}")
    
    if p_val_conc < 0.05:
        print("Result: Significant difference between concentrations (p < 0.05)")
    else:
        print("Result: No significant difference between concentrations (p >= 0.05)")

def correlation_analysis(df):
    """Analyze correlations"""
    print("\n=== CORRELATION ANALYSIS ===\n")
    
    # Correlation between concentration and inhibition zone
    correlation = df['Concentration_mg_mL'].corr(df['Inhibition_Zone_mm'])
    print(f"Correlation between Concentration and Inhibition Zone: {correlation:.4f}")
    
    # Statistical significance test
    r, p_value = stats.pearsonr(df['Concentration_mg_mL'], df['Inhibition_Zone_mm'])
    print(f"Pearson correlation coefficient: {r:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("Result: Significant correlation (p < 0.05)")
    else:
        print("Result: No significant correlation (p >= 0.05)")

def effectiveness_ranking(df):
    """Rank plant extracts by effectiveness"""
    print("\n=== EFFECTIVENESS RANKING ===\n")
    
    # Calculate mean inhibition zone for each plant
    plant_effectiveness = df.groupby('Plant_Extract')['Inhibition_Zone_mm'].agg([
        'mean', 'std', 'count'
    ]).round(2)
    
    # Calculate confidence intervals
    plant_effectiveness['ci_lower'] = plant_effectiveness['mean'] - 1.96 * (plant_effectiveness['std'] / np.sqrt(plant_effectiveness['count']))
    plant_effectiveness['ci_upper'] = plant_effectiveness['mean'] + 1.96 * (plant_effectiveness['std'] / np.sqrt(plant_effectiveness['count']))
    
    # Sort by mean effectiveness
    plant_effectiveness = plant_effectiveness.sort_values('mean', ascending=False)
    
    print("Plant Extract Effectiveness Ranking:")
    print("(Mean ± 95% Confidence Interval)")
    print("-" * 60)
    
    for i, (plant, stats) in enumerate(plant_effectiveness.iterrows(), 1):
        print(f"{i:2d}. {plant}")
        print(f"    Mean: {stats['mean']:.2f} mm")
        print(f"    95% CI: [{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]")
        print(f"    Std Dev: {stats['std']:.2f}")
        print()
    
    return plant_effectiveness

def main():
    """Main analysis function"""
    df = load_data()
    if df is None:
        return
    
    print("ANTIBACTERIAL ACTIVITY ANALYSIS")
    print("=" * 50)
    
    # Descriptive statistics
    plant_stats, bacteria_stats = descriptive_statistics(df)
    
    # ANOVA analysis
    anova_analysis(df)
    
    # Correlation analysis
    correlation_analysis(df)
    
    # Effectiveness ranking
    effectiveness_ranking(df)
    
    print("\n" + "=" * 50)
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()
