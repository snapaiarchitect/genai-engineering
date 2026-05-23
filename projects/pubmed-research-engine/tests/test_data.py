import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    errors = []
    
    try:
        assert os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'pubmed_abstracts.csv')), "Data file missing"
        print("✅ test_data_file_exists")
    except AssertionError as e:
        errors.append(f"❌ test_data_file_exists: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'pubmed_abstracts.csv'))
        assert len(df) >= 400, f"Expected >=400 abstracts, got {len(df)}"
        print(f"✅ test_data_has_records ({len(df)} abstracts)")
    except AssertionError as e:
        errors.append(f"❌ test_data_has_records: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'pubmed_abstracts.csv'))
        required = ['pmid', 'title', 'abstract', 'journal', 'year']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        print("✅ test_required_columns")
    except AssertionError as e:
        errors.append(f"❌ test_required_columns: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'pubmed_abstracts.csv'))
        assert df['title'].notna().all(), "All abstracts must have titles"
        assert df['abstract'].notna().sum() > 100, "Too few abstracts with text"
        print("✅ test_data_quality")
    except AssertionError as e:
        errors.append(f"❌ test_data_quality: {e}")
    
    try:
        figures_dir = os.path.join(PROJECT_ROOT, 'figures')
        required = [
            'publication_timeline.png', 'journal_distribution.png',
            'abstract_length_distribution.png', 'author_count_distribution.png',
            'mesh_terms.png', 'title_keywords.png'
        ]
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
