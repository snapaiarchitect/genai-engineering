import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    errors = []
    
    try:
        assert os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'corpus_stats.json')), "Data file missing"
        print("✅ test_data_file_exists")
    except AssertionError as e:
        errors.append(f"❌ test_data_file_exists: {e}")
    
    try:
        with open(os.path.join(PROJECT_ROOT, 'data', 'corpus_stats.json')) as f:
            data = json.load(f)
        assert data.get('total_docs', 0) > 100, f"Expected >100 docs, got {data.get('total_docs', 0)}"
        print(f"✅ test_data_has_records ({data.get('total_docs', 0)} docs)")
    except AssertionError as e:
        errors.append(f"❌ test_data_has_records: {e}")
    
    try:
        with open(os.path.join(PROJECT_ROOT, 'data', 'corpus_stats.json')) as f:
            data = json.load(f)
        assert 'top_categories' in data, "Corpus stats must have top_categories"
        print("✅ test_data_quality")
    except AssertionError as e:
        errors.append(f"❌ test_data_quality: {e}")
    
    try:
        figures_dir = os.path.join(PROJECT_ROOT, 'figures')
        required = ['tsne_embeddings.png', 'category_distribution.png']
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
