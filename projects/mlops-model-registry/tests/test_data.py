import pandas as pd
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    errors = []
    
    try:
        assert os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'acs_county_data.csv')), "Data file missing"
        print("✅ test_data_file_exists")
    except AssertionError as e:
        errors.append(f"❌ test_data_file_exists: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'acs_county_data.csv'))
        assert len(df) >= 3000, f"Expected >=3000 counties, got {len(df)}"
        print(f"✅ test_data_has_records ({len(df)} counties)")
    except AssertionError as e:
        errors.append(f"❌ test_data_has_records: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'acs_county_data.csv'))
        required = ['median_income', 'population', 'bachelors_degree', 'labor_force', 'median_rent']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        print("✅ test_required_columns")
    except AssertionError as e:
        errors.append(f"❌ test_required_columns: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'acs_county_data.csv'))
        assert (df['median_income'] > 0).sum() > 1000, "Too few valid income records"
        print("✅ test_data_quality")
    except AssertionError as e:
        errors.append(f"❌ test_data_quality: {e}")
    
    try:
        registry_dir = os.path.join(PROJECT_ROOT, 'models')
        assert os.path.exists(os.path.join(registry_dir, 'random_forest_v1.joblib')), "RF model missing"
        assert os.path.exists(os.path.join(registry_dir, 'ridge_v1.joblib')), "Ridge model missing"
        assert os.path.exists(os.path.join(registry_dir, 'registry_manifest.json')), "Manifest missing"
        print("✅ test_model_registry")
    except AssertionError as e:
        errors.append(f"❌ test_model_registry: {e}")
    
    try:
        figures_dir = os.path.join(PROJECT_ROOT, 'figures')
        required = ['01_registry_versions.png', '02_drift_detection.png', '03_ab_testing.png', '04_monitoring_overview.png']
        for fig in required:
            assert os.path.exists(os.path.join(figures_dir, fig)), f"Missing figure: {fig}"
        print("✅ test_figures_exist")
    except AssertionError as e:
        errors.append(f"❌ test_figures_exist: {e}")
    
    print(f"\n{'='*50}")
    if errors:
        print(f"FAILED: {len(errors)} test(s)")
        for e in errors:
            print(f"  {e}")
        return 1
    else:
        print("ALL TESTS PASSED ✅")
        return 0

if __name__ == '__main__':
    sys.exit(run_tests())
