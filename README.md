# Week 4: Advanced Analysis and Reporting of Language Trends using NLP

**Submission name:** Alisha Mondal  
**Project focus:** How technology-related vocabulary has changed in published English, alongside topic patterns in online discussion text.

## Project overview
This repository contains a reproducible NLP pipeline with two complementary public datasets:

1. **20 Newsgroups** — approximately 18,000 public newsgroup posts across 20 subject areas. The pipeline removes headers, footers and quoted replies, then performs tokenization, frequency analysis, TF-IDF, LDA, NMF and K-means clustering.
2. **Google Books Ngram Viewer (English 2019 corpus)** — yearly relative frequencies for `email`, `website`, `blog`, `podcast`, `hashtag`, `selfie`, `emoji` and `social media` from 1990 to 2019. This provides the time dimension for the language-trend analysis.

The two sources serve different purposes: 20 Newsgroups supports document-level topic discovery, while Google Books Ngrams supports historical trend detection. They should not be treated as one combined corpus.

## Repository structure
```text
Week4_Advanced_NLP_Language_Trends/
├── README.md
├── requirements.txt
├── src/
│   └── analyze.py
├── data/                 # downloaded/cached source data is not committed
├── outputs/              # generated CSV files, metrics and charts
└── report/
    └── Week4_NLP_Project_Report.docx
```

## Run on Windows
1. Install Python 3.10 or newer.
2. Download/clone this repository and open its folder in VS Code.
3. Create and activate a virtual environment:
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Run the complete pipeline:
   ```powershell
   python src/analyze.py
   ```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in that terminal, then activate the environment again. The first run needs internet access to download 20 Newsgroups and request Google Books Ngram data. Results are written to `outputs/`. This fresh starter ZIP contains an empty `outputs/` folder (only `.gitkeep` is present). No results from an earlier run are included. The CSV files, JSON summary and charts are created when the script finishes successfully.

## Generated outputs
- `cleaned_newsgroups_sample.csv` — cleaned document text and original category.
- `top_word_frequencies.csv` — most frequent content words.
- `nmf_topics.csv`, `lda_topics.csv` — topic terms.
- `cluster_sizes.csv` — K-means cluster counts.
- `ngram_trends.csv` — yearly relative frequencies.
- `run_summary.json` — dataset and model run metadata.
- `word_frequency.png`, `topic_distribution.png`, `language_trends.png` — charts.

## Methodology
- Normalize case, remove URLs and punctuation, and collapse whitespace.
- Remove English stop words during vectorization.
- Use CountVectorizer for LDA and TF-IDF n-grams for NMF and K-means.
- Use six topics/clusters as an interpretable exploratory setting, not as a claim that six is the objectively correct number.
- Fix random seeds for reproducibility. The K-means silhouette score is calculated on a bounded sample to keep runtime practical.
- Plot Google Books frequencies as relative frequencies; values are not raw counts.

## Important interpretation limits
Google Books represents published books, not all everyday speech or social-media language. The 20 Newsgroups collection is historical and domain-specific; it is not a representative sample of the modern internet. LDA/NMF topics require human interpretation, and K-means clusters can vary with preprocessing and parameter choices. Google Ngram values can change with corpus revisions and OCR improvements.

## References
- Scikit-learn, 20 Newsgroups dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
- Google Books Ngram Viewer information and corpus notes: https://books.google.com/ngrams/info
- Scikit-learn topic extraction example: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

## Reproducibility note
The pipeline downloads source data at run time and records model settings in `outputs/run_summary.json`. Do not report numerical findings until the script has completed successfully and the generated CSV files/charts have been reviewed.


## Before submission
Run the pipeline successfully and check that `outputs/run_summary.json`, all listed CSV files and all three PNG charts exist. The report is written to avoid claiming numerical findings before those outputs have been checked. Add a short results paragraph based on your own generated files if your instructor expects measured findings.
