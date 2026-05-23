import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    errors = []
    
    try:
        assert os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'scotus_cases.json')), "Data file missing"
        print("✅ test_data_file_exists")
    except AssertionError as e:
        errors.append(f"❌ test_data_file_exists: {e}")
    
    try:
        import json
        with open(os.path.join(PROJECT_ROOT, 'data', 'scotus_cases.json')) as f:
            data = json.load(f)
        assert len(data) >= 10, f"Expected >=10 cases, got {len(data)}"
        print(f"✅ test_data_has_records ({len(data)} cases)")
    except AssertionError as e:
        errors.append(f"❌ test_data_has_records: {e}")
    
    try:
        import json
        with open(os.path.join(PROJECT_ROOT, 'data', 'scotus_cases.json')) as f:
            data = json.load(f)
        assert all('case_name' in c for c in data), "All cases must have case_name"
        print("✅ test_data_quality")
    except AssertionError as e:
        errors.append(f"❌ test_data_quality: {e}")
    
    try:
        figures_dir = os.path.join(PROJECT_ROOT, 'figures')
        required = ['vote_margins.png', 'topic_distribution.png']
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
