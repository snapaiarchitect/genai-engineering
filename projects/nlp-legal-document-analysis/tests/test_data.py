import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    errors = []
    
    try:
        assert os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'congress_bills.csv')), "Data file missing"
        print("✅ test_data_file_exists")
    except AssertionError as e:
        errors.append(f"❌ test_data_file_exists: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'congress_bills.csv'))
        assert len(df) >= 400, f"Expected >=400 bills, got {len(df)}"
        print(f"✅ test_data_has_records ({len(df)} bills)")
    except AssertionError as e:
        errors.append(f"❌ test_data_has_records: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'congress_bills.csv'))
        required = ['congress', 'number', 'type', 'origin_chamber', 'title', 'latest_action_date']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        print("✅ test_required_columns")
    except AssertionError as e:
        errors.append(f"❌ test_required_columns: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'congress_bills.csv'))
        assert df['type'].notna().all(), "All bills must have a type"
        assert df['title'].notna().all(), "All bills must have a title"
        print("✅ test_data_quality")
    except AssertionError as e:
        errors.append(f"❌ test_data_quality: {e}")
    
    try:
        figures_dir = os.path.join(PROJECT_ROOT, 'figures')
        required_figs = [
            'bill_type_distribution.png', 'chamber_distribution.png', 'activity_timeline.png',
            'title_length_distribution.png', 'title_keywords.png', 'action_types.png'
        ]
        for fig in required_figs:
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
