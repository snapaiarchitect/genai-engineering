import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

def load_and_clean_data():
    """Load and clean ACS data."""
    df = pd.read_csv(get_path('data', 'acs_county_data.csv'))
    
    # Remove rows with missing or zero target
    df = df[df['median_income'] > 0].copy()
    
    # Feature engineering
    df['college_rate'] = df['bachelors_degree'] / df['population']
    df['labor_force_rate'] = df['labor_force'] / df['population']
    df['owner_rate'] = df['owner_occupied'] / (df['owner_occupied'] + df['renter_occupied'])
    df['pop_density_proxy'] = np.log1p(df['population'])
    
    # Select features
    features = ['population', 'college_rate', 'labor_force_rate', 'median_rent', 
                'commute_time', 'owner_rate', 'pop_density_proxy']
    target = 'median_income'
    
    # Drop rows with NaN in features
    df = df.dropna(subset=features + [target])
    
    return df, features, target

def train_and_register_models():
    """Train multiple models and save to registry."""
    df, features, target = load_and_clean_data()
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'random_forest_v1': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'ridge_v1': Ridge(alpha=1.0),
    }
    
    registry_dir = get_path('models')
    os.makedirs(registry_dir, exist_ok=True)
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'n_samples': len(y_test),
        }
        
        # Save model
        model_path = os.path.join(registry_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        
        # Save metadata
        meta = {
            'model_name': name,
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'target': target,
            'metrics': metrics,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'algorithm': type(model).__name__,
            'hyperparameters': model.get_params(),
        }
        
        meta_path = os.path.join(registry_dir, f"{name}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        results[name] = metrics
        print(f"  RMSE: ${metrics['rmse']:,.0f}, MAE: ${metrics['mae']:,.0f}, R²: {metrics['r2']:.3f}")
    
    # Save registry manifest
    manifest = {
        'registry_version': '1.0',
        'last_updated': datetime.now().isoformat(),
        'models': list(results.keys()),
        'best_model': min(results, key=lambda k: results[k]['rmse']),
        'all_metrics': results,
    }
    
    with open(os.path.join(registry_dir, 'registry_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nRegistry saved to {registry_dir}")
    print(f"Best model: {manifest['best_model']}")
    
    return results

if __name__ == '__main__':
    results = train_and_register_models()
