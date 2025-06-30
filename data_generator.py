import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_antibacterial_data():
    """Generate synthetic data for antibacterial activity testing"""
    
    # Plant extracts with varying effectiveness
    plant_extracts = [
        'Garlic (Allium sativum)',
        'Tea Tree (Melaleuca alternifolia)', 
        'Oregano (Origanum vulgare)',
        'Thyme (Thymus vulgaris)',
        'Eucalyptus (Eucalyptus globulus)',
        'Neem (Azadirachta indica)',
        'Turmeric (Curcuma longa)',
        'Ginger (Zingiber officinale)',
        'Cinnamon (Cinnamomum verum)',
        'Clove (Syzygium aromaticum)'
    ]
    
    # Common bacteria tested
    bacteria = [
        'Escherichia coli',
        'Staphylococcus aureus',
        'Streptococcus pyogenes',
        'Pseudomonas aeruginosa',
        'Bacillus subtilis',
        'Salmonella typhimurium',
        'Enterococcus faecalis',
        'Klebsiella pneumoniae'
    ]
    
    # Concentration levels (mg/mL)
    concentrations = [10, 25, 50, 100, 200]
    
    # Generate data
    data = []
    
    # Define effectiveness profiles for different plants
    plant_effectiveness = {
        'Garlic (Allium sativum)': {'base': 15, 'variance': 5},
        'Tea Tree (Melaleuca alternifolia)': {'base': 18, 'variance': 4},
        'Oregano (Origanum vulgare)': {'base': 16, 'variance': 3},
        'Thyme (Thymus vulgaris)': {'base': 17, 'variance': 4},
        'Eucalyptus (Eucalyptus globulus)': {'base': 12, 'variance': 6},
        'Neem (Azadirachta indica)': {'base': 14, 'variance': 5},
        'Turmeric (Curcuma longa)': {'base': 10, 'variance': 4},
        'Ginger (Zingiber officinale)': {'base': 8, 'variance': 3},
        'Cinnamon (Cinnamomum verum)': {'base': 13, 'variance': 4},
        'Clove (Syzygium aromaticum)': {'base': 19, 'variance': 3}
    }
    
    # Bacterial resistance factors
    bacterial_resistance = {
        'Escherichia coli': 1.0,
        'Staphylococcus aureus': 0.8,
        'Streptococcus pyogenes': 1.2,
        'Pseudomonas aeruginosa': 0.6,
        'Bacillus subtilis': 1.4,
        'Salmonella typhimurium': 0.9,
        'Enterococcus faecalis': 0.7,
        'Klebsiella pneumoniae': 0.8
    }
    
    for plant in plant_extracts:
        for bacterium in bacteria:
            for concentration in concentrations:
                for replicate in range(3):  # 3 replicates per condition
                    
                    # Calculate zone of inhibition (mm)
                    base_effectiveness = plant_effectiveness[plant]['base']
                    variance = plant_effectiveness[plant]['variance']
                    resistance_factor = bacterial_resistance[bacterium]
                    
                    # Concentration effect (logarithmic relationship)
                    conc_effect = np.log10(concentration / 10) * 3
                    
                    # Calculate inhibition zone
                    inhibition_zone = (base_effectiveness + conc_effect) * resistance_factor
                    inhibition_zone += np.random.normal(0, variance)
                    
                    # Ensure minimum of 0
                    inhibition_zone = max(0, inhibition_zone)
                    
                    # Determine activity level
                    if inhibition_zone >= 20:
                        activity_level = 'High'
                    elif inhibition_zone >= 15:
                        activity_level = 'Moderate'
                    elif inhibition_zone >= 10:
                        activity_level = 'Low'
                    else:
                        activity_level = 'None'
                    
                    data.append({
                        'Plant_Extract': plant,
                        'Bacteria': bacterium,
                        'Concentration_mg_mL': concentration,
                        'Replicate': replicate + 1,
                        'Inhibition_Zone_mm': round(inhibition_zone, 2),
                        'Activity_Level': activity_level,
                        'Test_Date': datetime.now() - timedelta(days=random.randint(1, 30))
                    })
    
    df = pd.DataFrame(data)
    df.to_csv('antibacterial_data.csv', index=False)
    print(f"Generated {len(df)} data points")
    print(f"Data saved to 'antibacterial_data.csv'")
    return df

if __name__ == "__main__":
    df = generate_antibacterial_data()
    print("\nData Preview:")
    print(df.head(10))
    print(f"\nData shape: {df.shape}")
