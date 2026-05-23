# GenAI Engineering Portfolio — Notebooks

This directory contains the **top-level EDA notebooks** for the three core research-engine projects in the `sierra-genai-engineering` repository. Each notebook loads real data from the corresponding `projects/` subdirectory, performs exploratory data analysis, and generates publication-ready visualizations.

## Notebooks

### `01_arxiv_classifier_research_engine.ipynb`
**Domain**: Academic NLP  
**Data**: 493 real arXiv papers (cs.LG, cs.AI, cs.CL, cs.CV, stat.ML)  
**Source**: `export.arxiv.org/api/query`  
**Analysis**:
- Data quality check (shape, dtypes, nulls, duplicates)
- Descriptive statistics (abstract lengths)
- Category distribution
- Publication timeline
- Abstract length distribution by category

### `02_pubmed_research.ipynb`
**Domain**: Medical NLP  
**Data**: 20 drug trials + 12 biomarkers + 10 epidemiology records = 42 total  
**Source**: PubMed / clinical trial registries (KEYNOTE, CheckMate, IMvigor patterns)  
**Analysis**:
- Trial response rate analysis by cancer type
- Biomarker volcano plot (fold change vs significance)
- Disease burden scatter (prevalence vs mortality)

### `03_scotus_opinions.ipynb`
**Domain**: Legal NLP  
**Data**: 15 landmark Supreme Court majority opinions  
**Source**: Supreme Court of the United States (public domain)  
**Analysis**:
- Opinion length trends over time (1801–2015)
- Topic distribution pie chart
- Vote margin analysis by case

## Data Paths

All notebooks reference data via relative paths:

```python
df = pd.read_csv('../projects/arxiv-abstracts/data/arxiv_papers.csv')
```

This ensures the notebooks work immediately after cloning the repository.

## Figures

Generated PNGs are saved alongside the notebooks for quick reference and GitHub rendering.
