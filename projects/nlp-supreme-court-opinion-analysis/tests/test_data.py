import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    errors = []
    
    try:
        assert os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'scotus_cases.csv')), "Data file missing"
        print("✅ test_data_file_exists")
    except AssertionError as e:
        errors.append(f"❌ test_data_file_exists: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'scotus_cases.csv'))
        assert len(df) >= 50, f"Expected >=50 cases, got {len(df)}"
        print(f"✅ test_data_has_records ({len(df)} cases)")
    except AssertionError as e:
        errors.append(f"❌ test_data_has_records: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'scotus_cases.csv'))
        required = ['name', 'docket_number', 'majority_votes', 'minority_votes', 'date_argued', 'date_decided']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        print("✅ test_required_columns")
    except AssertionError as e:
        errors.append(f"❌ test_required_columns: {e}")
    
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'scotus_cases.csv'))
        vote_cols = [c for c in df.columns if c.startswith('vote_')]
        assert len(vote_cols) >= 5, f"Expected >=5 justice vote columns, got {len(vote_cols)}"
        print(f"✅ test_justice_votes ({len(vote_cols)} justices)")
    except AssertionError as e:
        errors.append(f"❌ test_justice_votes: {e}")
    
    try:
        figures_dir = os.path.join(PROJECT_ROOT, 'figures')
        required = [
            'vote_margin_distribution.png', 'justice_alignment_matrix.png',
            'opinion_type_distribution.png', 'case_timeline.png',
            'majority_vote_distribution.png', 'case_topic_keywords.png'
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
