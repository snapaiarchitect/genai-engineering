import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

def run_tests():
    errors = []
    
    # Test 1: Data file exists
    try:
        assert os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'clinical_trials.csv')), "Data file missing"
        print("✅ test_data_file_exists")
    except AssertionError as e:
        errors.append(f"❌ test_data_file_exists: {e}")
    
    # Test 2: Data has records
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'clinical_trials.csv'))
        assert len(df) >= 400, f"Expected >=400 trials, got {len(df)}"
        print(f"✅ test_data_has_records ({len(df)} trials)")
    except AssertionError as e:
        errors.append(f"❌ test_data_has_records: {e}")
    
    # Test 3: Required columns
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'clinical_trials.csv'))
        required = ['nct_id', 'title', 'condition', 'phase', 'sponsor_class', 'enrollment_count']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        print("✅ test_required_columns")
    except AssertionError as e:
        errors.append(f"❌ test_required_columns: {e}")
    
    # Test 4: NCT IDs valid
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'clinical_trials.csv'))
        assert df['nct_id'].str.startswith('NCT').all(), "All NCT IDs must start with NCT"
        assert df['nct_id'].str.len().eq(11).all(), "NCT IDs must be 11 characters"
        print("✅ test_nct_ids_are_valid")
    except AssertionError as e:
        errors.append(f"❌ test_nct_ids_are_valid: {e}")
    
    # Test 5: Figures exist
    try:
        figures_dir = os.path.join(PROJECT_ROOT, 'figures')
        required_figs = [
            'phase_distribution.png', 'sponsor_distribution.png', 'enrollment_analysis.png',
            'condition_frequency.png', 'study_intervention.png', 'geographic_spread.png'
        ]
        for fig in required_figs:
            assert os.path.exists(os.path.join(figures_dir, fig)), f"Missing figure: {fig}"
        print("✅ test_figures_exist")
    except AssertionError as e:
        errors.append(f"❌ test_figures_exist: {e}")
    
    # Test 6: Phases valid
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'clinical_trials.csv'))
        valid_phases = ['PHASE1', 'PHASE2', 'PHASE3', 'PHASE4', 'EARLY_PHASE1', 'NA', 
                        'PHASE1, PHASE2', 'PHASE2, PHASE3']
        phases = df['phase'].dropna().replace('', pd.NA).dropna()
        for phase in phases.unique():
            assert phase in valid_phases, f"Unexpected phase: {phase}"
        print("✅ test_phases_are_valid")
    except AssertionError as e:
        errors.append(f"❌ test_phases_are_valid: {e}")
    
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
