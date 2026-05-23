import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    errors = []
    
    try:
        assert os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'arxiv_papers.csv')), "Data file missing"
        print("✅ test_data_file_exists")
    except AssertionError as e:
        errors.append(f"❌ test_data_file_exists: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'arxiv_papers.csv'))
        assert len(df) >= 400, f"Expected >=400 papers, got {len(df)}"
        print(f"✅ test_data_has_records ({len(df)} papers)")
    except AssertionError as e:
        errors.append(f"❌ test_data_has_records: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'arxiv_papers.csv'))
        required = ['title', 'abstract', 'primary_category']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        print("✅ test_required_columns")
    except AssertionError as e:
        errors.append(f"❌ test_required_columns: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'arxiv_papers.csv'))
        assert df['title'].notna().all(), "All papers must have titles"
        print("✅ test_data_quality")
    except AssertionError as e:
        errors.append(f"❌ test_data_quality: {e}")
    
    try:
        figures_dir = os.path.join(PROJECT_ROOT, 'figures')
        required = ['category_distribution.png', 'publication_timeline.png', 'top_keywords.png']
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
