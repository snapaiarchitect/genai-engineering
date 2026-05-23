import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    errors = []
    
    try:
        assert os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_documents.parquet')), "Data file missing"
        print("✅ test_data_file_exists")
    except AssertionError as e:
        errors.append(f"❌ test_data_file_exists: {e}")
    
    try:
        import pandas as pd
        df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_documents.parquet'))
        assert len(df) >= 400, f"Expected >=400 docs, got {len(df)}"
        print(f"✅ test_data_has_records ({len(df)} docs)")
    except AssertionError as e:
        errors.append(f"❌ test_data_has_records: {e}")
    
    try:
        import pandas as pd
        df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_documents.parquet'))
        assert 'category' in df.columns, "Missing category column"
        assert 'text' in df.columns, "Missing text column"
        print("✅ test_required_columns")
    except AssertionError as e:
        errors.append(f"❌ test_required_columns: {e}")
    
    try:
        figures_dir = os.path.join(PROJECT_ROOT, 'figures')
        pngs = [f for f in os.listdir(figures_dir) if f.endswith('.png')] if os.path.exists(figures_dir) else []
        assert len(pngs) >= 2, f"Expected >=2 figures, got {len(pngs)}"
        print(f"✅ test_figures_exist ({len(pngs)} figures)")
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
