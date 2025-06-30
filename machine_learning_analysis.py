import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for machine learning"""
    try:
        df = pd.read_csv('antibacterial_data.csv')
    except FileNotFoundError:
        print("Data file not found. Please run data_generator.py first.")
        return None, None
    
    # Prepare features
    # Encode categorical variables
    le_plant = LabelEncoder()
    le_bacteria = LabelEncoder()
    
    df['Plant_Encoded'] = le_plant.fit_transform(df['Plant_Extract'])
    df['Bacteria_Encoded'] = le_bacteria.fit_transform(df['Bacteria'])
    
    # Features for prediction
    features = ['Plant_Encoded', 'Bacteria_Encoded', 'Concentration_mg_mL']
    X = df[features]
    y = df['Inhibition_Zone_mm']
    
    # Store encoders for later use
    encoders = {
        'plant': le_plant,
        'bacteria': le_bacteria
    }
    
    return X, y, df, encoders

def train_models(X, y):
    """Train multiple machine learning models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("=== MACHINE LEARNING MODEL PERFORMANCE ===\n")
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Use scaled data for Linear Regression, original for tree-based models
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            X_cv = X_train_scaled
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            X_cv = X_train
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_cv, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  R²: {r2:.3f}")
        print(f"  CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print()
    
    return results, scaler

def feature_importance_analysis(results):
    """Analyze feature importance"""
    print("=== FEATURE IMPORTANCE ANALYSIS ===\n")
    
    # Get Random Forest feature importance
    rf_model = results['Random Forest']['model']
    feature_names = ['Plant Extract', 'Bacteria', 'Concentration']
    importance = rf_model.feature_importances_
    
    print("Random Forest Feature Importance:")
    for name, imp in zip(feature_names, importance):
        print(f"  {name}: {imp:.3f}")
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    bars = plt.bar(feature_names, importance, color=['lightcoral', 'lightblue', 'lightgreen'])
    plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    plt.ylabel('Importance')
    plt.ylim(0, max(importance) * 1.1)
    
    # Add value labels on bars
    for bar, imp in zip(bars, importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{imp:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def model_comparison_plot(results):
    """Create model comparison visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance metrics comparison
    models = list(results.keys())
    metrics = ['RMSE', 'MAE', 'R²']
    
    rmse_values = [results[model]['rmse'] for model in models]
    mae_values = [results[model]['mae'] for model in models]
    r2_values = [results[model]['r2'] for model in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    axes[0].bar(x - width, rmse_values, width, label='RMSE', alpha=0.8)
    axes[0].bar(x, mae_values, width, label='MAE', alpha=0.8)
    axes[0].bar(x + width, r2_values, width, label='R²', alpha=0.8)
    
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Prediction vs Actual scatter plot for best model
    best_model = max(results.keys(), key=lambda k: results[k]['r2'])
    y_test = results[best_model]['y_test']
    y_pred = results[best_model]['y_pred']
    
    axes[1].scatter(y_test, y_pred, alpha=0.6, color='blue')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Inhibition Zone (mm)')
    axes[1].set_ylabel('Predicted Inhibition Zone (mm)')
    axes[1].set_title(f'Prediction vs Actual ({best_model})')
    axes[1].grid(True, alpha=0.3)
    
    # Add R² to the plot
    r2 = results[best_model]['r2']
    axes[1].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def predict_new_combinations(results, encoders, df):
    """Predict antibacterial activity for new combinations"""
    print("\n=== PREDICTIONS FOR NEW COMBINATIONS ===\n")
    
    # Get the best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"Using {best_model_name} for predictions (R² = {results[best_model_name]['r2']:.3f})")
    print()
    
    # Example predictions
    plant_names = df['Plant_Extract'].unique()
    bacteria_names = df['Bacteria'].unique()
    
    # Select some interesting combinations
    test_combinations = [
        ('Clove (Syzygium aromaticum)', 'Pseudomonas aeruginosa', 150),
        ('Tea Tree (Melaleuca alternifolia)', 'Staphylococcus aureus', 75),
        ('Garlic (Allium sativum)', 'Escherichia coli', 125),
        ('Oregano (Origanum vulgare)', 'Salmonella typhimurium', 100),
        ('Neem (Azadirachta indica)', 'Klebsiella pneumoniae', 200)
    ]
    
    print("Predicted Inhibition Zones for New Combinations:")
    print("-" * 60)
    
    for plant, bacteria, concentration in test_combinations:
        # Encode the inputs
        plant_encoded = encoders['plant'].transform([plant])[0]
        bacteria_encoded = encoders['bacteria'].transform([bacteria])[0]
        
        # Make prediction
        X_new = np.array([[plant_encoded, bacteria_encoded, concentration]])
        prediction = best_model.predict(X_new)[0]
        
        # Determine activity level
        if prediction >= 20:
            activity = 'High'
        elif prediction >= 15:
            activity = 'Moderate'
        elif prediction >= 10:
            activity = 'Low'
        else:
            activity = 'None'
        
        print(f"Plant: {plant.split('(')[0].strip()}")
        print(f"Bacteria: {bacteria}")
        print(f"Concentration: {concentration} mg/mL")
        print(f"Predicted Inhibition Zone: {prediction:.2f} mm")
        print(f"Predicted Activity Level: {activity}")
        print("-" * 60)

def main():
    """Main machine learning analysis function"""
    print("MACHINE LEARNING ANALYSIS")
    print("=" * 50)
    
    # Load and prepare data
    X, y, df, encoders = load_and_prepare_data()
    if X is None:
        return
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print()
    
    # Train models
    results, scaler = train_models(X, y)
    
    # Feature importance analysis
    feature_importance_analysis(results)
    
    # Model comparison
    model_comparison_plot(results)
    
    # Predictions for new combinations
    predict_new_combinations(results, encoders, df)
    
    print("\n" + "=" * 50)
    print("Machine learning analysis completed!")
    print("Files saved:")
    print("- feature_importance.png")
    print("- model_comparison.png")

if __name__ == "__main__":
    main()
