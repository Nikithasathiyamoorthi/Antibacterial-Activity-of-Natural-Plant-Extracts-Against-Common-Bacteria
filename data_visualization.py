import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the antibacterial data"""
    try:
        df = pd.read_csv('antibacterial_data.csv')
        return df
    except FileNotFoundError:
        print("Data file not found. Please run data_generator.py first.")
        return None

def create_overview_plots(df):
    """Create overview visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Antibacterial Activity Overview', fontsize=16, fontweight='bold')
    
    # 1. Distribution of inhibition zones
    axes[0, 0].hist(df['Inhibition_Zone_mm'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Inhibition Zones')
    axes[0, 0].set_xlabel('Inhibition Zone (mm)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Activity level distribution
    activity_counts = df['Activity_Level'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    axes[0, 1].pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 1].set_title('Activity Level Distribution')
    
    # 3. Concentration vs Inhibition Zone
    concentrations = df['Concentration_mg_mL'].unique()
    mean_inhibition = [df[df['Concentration_mg_mL'] == c]['Inhibition_Zone_mm'].mean() for c in concentrations]
    std_inhibition = [df[df['Concentration_mg_mL'] == c]['Inhibition_Zone_mm'].std() for c in concentrations]
    
    axes[1, 0].errorbar(concentrations, mean_inhibition, yerr=std_inhibition, 
                        marker='o', capsize=5, capthick=2, linewidth=2)
    axes[1, 0].set_title('Concentration vs Mean Inhibition Zone')
    axes[1, 0].set_xlabel('Concentration (mg/mL)')
    axes[1, 0].set_ylabel('Mean Inhibition Zone (mm)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Top 5 most effective plants
    plant_means = df.groupby('Plant_Extract')['Inhibition_Zone_mm'].mean().sort_values(ascending=True).tail(5)
    axes[1, 1].barh(range(len(plant_means)), plant_means.values, color='lightgreen')
    axes[1, 1].set_yticks(range(len(plant_means)))
    axes[1, 1].set_yticklabels([name.split('(')[0].strip() for name in plant_means.index])
    axes[1, 1].set_title('Top 5 Most Effective Plant Extracts')
    axes[1, 1].set_xlabel('Mean Inhibition Zone (mm)')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('overview_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_heatmap(df):
    """Create heatmap of plant extracts vs bacteria"""
    # Calculate mean inhibition zones
    heatmap_data = df.groupby(['Plant_Extract', 'Bacteria'])['Inhibition_Zone_mm'].mean().unstack()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                cbar_kws={'label': 'Mean Inhibition Zone (mm)'})
    plt.title('Antibacterial Activity Heatmap\n(Plant Extracts vs Bacteria)', fontsize=14, fontweight='bold')
    plt.xlabel('Bacteria Species')
    plt.ylabel('Plant Extracts')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Simplify plant names for better readability
    y_labels = [label.get_text().split('(')[0].strip() for label in plt.gca().get_yticklabels()]
    plt.gca().set_yticklabels(y_labels)
    
    plt.tight_layout()
    plt.savefig('activity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_boxplots(df):
    """Create box plots for detailed analysis"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Box plot by plant extract
    plant_order = df.groupby('Plant_Extract')['Inhibition_Zone_mm'].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Plant_Extract', y='Inhibition_Zone_mm', order=plant_order, ax=axes[0])
    axes[0].set_title('Inhibition Zone Distribution by Plant Extract', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Plant Extract')
    axes[0].set_ylabel('Inhibition Zone (mm)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Simplify x-axis labels
    x_labels = [label.get_text().split('(')[0].strip() for label in axes[0].get_xticklabels()]
    axes[0].set_xticklabels(x_labels)
    
    # Box plot by bacteria
    bacteria_order = df.groupby('Bacteria')['Inhibition_Zone_mm'].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Bacteria', y='Inhibition_Zone_mm', order=bacteria_order, ax=axes[1])
    axes[1].set_title('Inhibition Zone Distribution by Bacteria', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Bacteria Species')
    axes[1].set_ylabel('Inhibition Zone (mm)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('distribution_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_concentration_analysis(df):
    """Create concentration effect analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Line plot showing concentration effect for top 5 plants
    top_plants = df.groupby('Plant_Extract')['Inhibition_Zone_mm'].mean().nlargest(5).index
    
    for plant in top_plants:
        plant_data = df[df['Plant_Extract'] == plant]
        conc_means = plant_data.groupby('Concentration_mg_mL')['Inhibition_Zone_mm'].mean()
        axes[0].plot(conc_means.index, conc_means.values, marker='o', linewidth=2, 
                    label=plant.split('(')[0].strip())
    
    axes[0].set_title('Concentration Effect - Top 5 Plant Extracts', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Concentration (mg/mL)')
    axes[0].set_ylabel('Mean Inhibition Zone (mm)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Violin plot by concentration
    sns.violinplot(data=df, x='Concentration_mg_mL', y='Inhibition_Zone_mm', ax=axes[1])
    axes[1].set_title('Inhibition Zone Distribution by Concentration', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Concentration (mg/mL)')
    axes[1].set_ylabel('Inhibition Zone (mm)')
    
    plt.tight_layout()
    plt.savefig('concentration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_effectiveness_chart(df):
    """Create effectiveness ranking chart"""
    # Calculate effectiveness metrics
    plant_stats = df.groupby('Plant_Extract').agg({
        'Inhibition_Zone_mm': ['mean', 'std', 'count']
    }).round(2)
    
    plant_stats.columns = ['mean', 'std', 'count']
    plant_stats['se'] = plant_stats['std'] / np.sqrt(plant_stats['count'])
    plant_stats = plant_stats.sort_values('mean', ascending=True)
    
    # Create horizontal bar chart with error bars
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(plant_stats))
    bars = ax.barh(y_pos, plant_stats['mean'], xerr=plant_stats['se'], 
                   capsize=5, alpha=0.7, color='lightcoral')
    
    # Color bars based on effectiveness
    for i, (bar, mean_val) in enumerate(zip(bars, plant_stats['mean'])):
        if mean_val >= 18:
            bar.set_color('darkgreen')
        elif mean_val >= 15:
            bar.set_color('green')
        elif mean_val >= 12:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.split('(')[0].strip() for name in plant_stats.index])
    ax.set_xlabel('Mean Inhibition Zone (mm)')
    ax.set_title('Plant Extract Effectiveness Ranking\n(with Standard Error)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add effectiveness legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='darkgreen', label='High (≥18mm)'),
        Rectangle((0, 0), 1, 1, facecolor='green', label='Moderate (15-18mm)'),
        Rectangle((0, 0), 1, 1, facecolor='orange', label='Low (12-15mm)'),
        Rectangle((0, 0), 1, 1, facecolor='red', label='Minimal (<12mm)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('effectiveness_ranking.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main visualization function"""
    df = load_data()
    if df is None:
        return
    
    print("Creating visualizations...")
    print("1. Overview plots...")
    create_overview_plots(df)
    
    print("2. Activity heatmap...")
    create_heatmap(df)
    
    print("3. Distribution box plots...")
    create_boxplots(df)
    
    print("4. Concentration analysis...")
    create_concentration_analysis(df)
    
    print("5. Effectiveness ranking...")
    create_effectiveness_chart(df)
    
    print("\nAll visualizations created and saved!")
    print("Files saved:")
    print("- overview_plots.png")
    print("- activity_heatmap.png") 
    print("- distribution_boxplots.png")
    print("- concentration_analysis.png")
    print("- effectiveness_ranking.png")

if __name__ == "__main__":
    main()
