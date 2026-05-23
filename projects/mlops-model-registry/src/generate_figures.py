import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

os.makedirs(get_path('figures'), exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Registry Versions
fig, ax = plt.subplots(figsize=(10, 6))
registry_dir = get_path('models')
models = ['random_forest_v1', 'ridge_v1']
metrics = []
for m in models:
    meta_path = os.path.join(registry_dir, f"{m}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        metrics.append({
            'model': m,
            'rmse': meta['metrics']['rmse'],
            'mae': meta['metrics']['mae'],
            'r2': meta['metrics']['r2'],
        })

metrics_df = pd.DataFrame(metrics)
x = np.arange(len(metrics_df))
width = 0.25

ax.bar(x - width, metrics_df['rmse']/1000, width, label='RMSE ($K)', color='coral')
ax.bar(x, metrics_df['mae']/1000, width, label='MAE ($K)', color='steelblue')
ax.bar(x + width, metrics_df['r2']*10000, width, label='R² (×10K)', color='seagreen')

ax.set_xticks(x)
ax.set_xticklabels(metrics_df['model'], rotation=15, ha='right')
ax.set_ylabel('Score')
ax.set_title('Model Registry: Version Comparison', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(get_path('figures', '01_registry_versions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("1. Registry versions saved")

# 2. Drift Detection — simulate by comparing two random splits
df = pd.read_csv(get_path('data', 'acs_county_data.csv'))
df = df[df['median_income'] > 0]
features = ['population', 'bachelors_degree', 'labor_force', 'median_rent', 'commute_time']

# Split into two batches
batch_a = df.sample(frac=0.5, random_state=42)
batch_b = df.drop(batch_a.index)

def calculate_psi(expected, actual, buckets=10):
    """Population Stability Index."""
    expected_percents = np.histogram(expected, bins=buckets, density=True)[0]
    actual_percents = np.histogram(actual, bins=buckets, density=True)[0]
    
    # Add small constant to avoid division by zero
    expected_percents = np.clip(expected_percents, 0.0001, 1)
    actual_percents = np.clip(actual_percents, 0.0001, 1)
    
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi

psi_scores = {}
for feat in features:
    psi = calculate_psi(batch_a[feat].dropna(), batch_b[feat].dropna())
    psi_scores[feat] = psi

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['seagreen' if v < 0.1 else 'gold' if v < 0.25 else 'crimson' for v in psi_scores.values()]
bars = ax.barh(list(psi_scores.keys()), list(psi_scores.values()), color=colors)
ax.axvline(0.1, color='green', linestyle='--', label='Stable (<0.1)')
ax.axvline(0.25, color='red', linestyle='--', label='Significant (>0.25)')
ax.set_xlabel('Population Stability Index (PSI)')
ax.set_title('Feature Drift Detection (Batch A vs Batch B)', fontsize=14, fontweight='bold')
ax.legend()
for bar, val in zip(bars, psi_scores.values()):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(get_path('figures', '02_drift_detection.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"2. Drift detection saved. PSI scores: {psi_scores}")

# 3. A/B Testing — compare model predictions on holdout
model_a = joblib.load(os.path.join(registry_dir, 'random_forest_v1.joblib'))
model_b = joblib.load(os.path.join(registry_dir, 'ridge_v1.joblib'))

test_df = df.sample(frac=0.1, random_state=123)
# Recalculate engineered features
test_df['college_rate'] = test_df['bachelors_degree'] / test_df['population']
test_df['labor_force_rate'] = test_df['labor_force'] / test_df['population']
test_df['owner_rate'] = test_df['owner_occupied'] / (test_df['owner_occupied'] + test_df['renter_occupied'])
test_df['pop_density_proxy'] = np.log1p(test_df['population'])

X_test = test_df[['population', 'college_rate', 'labor_force_rate', 'median_rent', 'commute_time', 'owner_rate', 'pop_density_proxy']].fillna(0)
y_test = test_df['median_income']

pred_a = model_a.predict(X_test)
pred_b = model_b.predict(X_test)

errors_a = np.abs(y_test - pred_a)
errors_b = np.abs(y_test - pred_b)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test, pred_a, alpha=0.3, label='Random Forest', color='coral')
axes[0].scatter(y_test, pred_b, alpha=0.3, label='Ridge', color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect')
axes[0].set_xlabel('Actual Income ($)')
axes[0].set_ylabel('Predicted Income ($)')
axes[0].set_title('A/B Test: Prediction vs Actual')
axes[0].legend()

axes[1].hist(errors_a, bins=30, alpha=0.5, label=f'RF (MAE=${errors_a.mean():,.0f})', color='coral')
axes[1].hist(errors_b, bins=30, alpha=0.5, label=f'Ridge (MAE=${errors_b.mean():,.0f})', color='steelblue')
axes[1].set_xlabel('Absolute Error ($)')
axes[1].set_ylabel('Count')
axes[1].set_title('A/B Test: Error Distribution')
axes[1].legend()

plt.tight_layout()
plt.savefig(get_path('figures', '03_ab_testing.png'), dpi=150, bbox_inches='tight')
plt.close()
print("3. A/B testing saved")

# 4. Monitoring Overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Feature importance
rf_model = model_a
importances = rf_model.feature_importances_
feature_names = ['population', 'college_rate', 'labor_force_rate', 'median_rent', 'commute_time', 'owner_rate', 'pop_density_proxy']
feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance')
axes[0,0].barh(feat_df['feature'], feat_df['importance'], color='steelblue')
axes[0,0].set_title('Feature Importance (Random Forest)')

# Residual distribution
residuals = y_test - pred_a
axes[0,1].hist(residuals, bins=40, color='coral', edgecolor='white')
axes[0,1].axvline(0, color='black', linestyle='--')
axes[0,1].set_title(f'Residual Distribution (μ={residuals.mean():,.0f}, σ={residuals.std():,.0f})')
axes[0,1].set_xlabel('Residual ($)')

# Income distribution
axes[1,0].hist(df['median_income'].dropna(), bins=50, color='seagreen', edgecolor='white')
axes[1,0].axvline(df['median_income'].median(), color='red', linestyle='--', label=f'Median: ${df["median_income"].median():,.0f}')
axes[1,0].set_title('County Median Income Distribution')
axes[1,0].set_xlabel('Median Household Income ($)')
axes[1,0].legend()

# Model metrics over time (simulated)
metrics_history = {
    'week': ['W1', 'W2', 'W3', 'W4'],
    'rmse': [9500, 9200, 8911, 8800],
    'r2': [0.72, 0.74, 0.756, 0.76],
}
axes[1,1].plot(metrics_history['week'], metrics_history['rmse'], marker='o', color='coral', label='RMSE')
ax2 = axes[1,1].twinx()
ax2.plot(metrics_history['week'], metrics_history['r2'], marker='s', color='steelblue', label='R²')
axes[1,1].set_ylabel('RMSE ($)', color='coral')
ax2.set_ylabel('R²', color='steelblue')
axes[1,1].set_title('Model Performance Monitoring (Simulated)')
axes[1,1].legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(get_path('figures', '04_monitoring_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("4. Monitoring overview saved")

print("\nAll figures generated successfully")
