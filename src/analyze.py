"""Reproducible language-trend analysis using 20 Newsgroups and Google Books Ngrams."""
from pathlib import Path
import json, re, sys, time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
TERMS = ["email", "website", "blog", "podcast", "hashtag", "selfie", "emoji", "social media"]


def clean_text(text):
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text).lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_newsgroups():
    # Remove headers, signatures and quoted replies to reduce metadata leakage.
    ds = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"),
                            shuffle=True, random_state=SEED)
    frame = pd.DataFrame({"text": ds.data, "category": [ds.target_names[i] for i in ds.target]})
    frame["clean_text"] = frame.text.map(clean_text)
    frame = frame[frame.clean_text.str.len() > 80].reset_index(drop=True)
    frame.to_csv(OUT / "cleaned_newsgroups_sample.csv", index=False)
    return frame


def topic_models(frame):
    docs = frame.clean_text.tolist()
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.85, min_df=5,
                            ngram_range=(1, 2), max_features=12000)
    X = tfidf.fit_transform(docs)
    lda_vec = CountVectorizer(stop_words="english", max_df=0.85, min_df=5,
                               max_features=12000)
    X_count = lda_vec.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=6, random_state=SEED,
                                    learning_method="batch", max_iter=10)
    lda.fit(X_count)
    nmf = NMF(n_components=6, random_state=SEED, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    terms = np.array(tfidf.get_feature_names_out())
    rows = []
    for i, topic in enumerate(nmf.components_):
        top = terms[topic.argsort()[-12:][::-1]].tolist()
        rows.append({"topic": i + 1, "top_terms": ", ".join(top),
                     "documents_dominant": int((W.argmax(axis=1) == i).sum())})
    pd.DataFrame(rows).to_csv(OUT / "nmf_topics.csv", index=False)
    lda_terms = np.array(lda_vec.get_feature_names_out())
    lda_rows = []
    for i, topic in enumerate(lda.components_):
        lda_rows.append({"topic": i + 1, "top_terms": ", ".join(lda_terms[topic.argsort()[-12:][::-1]])})
    pd.DataFrame(lda_rows).to_csv(OUT / "lda_topics.csv", index=False)
    # Cluster the document vectors and report silhouette on a reproducible subset.
    km = KMeans(n_clusters=6, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    sample_n = min(2500, X.shape[0])
    # Use a seeded random sample rather than only the first rows.
    rng = np.random.default_rng(SEED)
    sample_idx = np.sort(rng.choice(X.shape[0], size=sample_n, replace=False))
    sample_labels = labels[sample_idx]
    sil = silhouette_score(X[sample_idx], sample_labels) if len(set(sample_labels)) > 1 else float("nan")
    frame.assign(cluster=labels).groupby("cluster").size().rename("documents").to_csv(OUT / "cluster_sizes.csv")
    return {"documents_used": len(frame), "vocabulary_size": len(terms),
            "nmf_topics": 6, "lda_topics": 6, "kmeans_clusters": 6,
            "silhouette_sample": sample_n, "silhouette_score": float(sil)}


def fetch_ngram_trends():
    url = "https://books.google.com/ngrams/json"
    params = {"content": ",".join(TERMS), "year_start": 1990, "year_end": 2019,
              "corpus": "en-2019", "smoothing": 3}
    response = requests.get(url, params=params, timeout=60,
                            headers={"User-Agent": "Week4-NLP-Student-Research/1.0"})
    response.raise_for_status()
    payload = response.json()
    if not payload:
        raise RuntimeError("Google Books Ngram API returned no series. Check terms or endpoint.")
    rows = []
    expected_years = list(range(1990, 2020))
    returned_terms = {item.get("ngram") for item in payload}
    missing_terms = sorted(set(TERMS) - returned_terms)
    if missing_terms:
        raise RuntimeError(f"Google Books Ngram response is missing terms: {missing_terms}")
    for item in payload:
        values = item.get("timeseries", [])
        if len(values) != len(expected_years):
            raise RuntimeError(
                f"Unexpected yearly series length for {item.get('ngram')}: "
                f"expected {len(expected_years)}, received {len(values)}"
            )
        for year, value in zip(expected_years, values):
            rows.append({"term": item["ngram"], "year": year, "relative_frequency": value})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ngram_trends.csv", index=False)
    return df


def make_charts(frame, trends):
    # Top unigram frequency chart
    cv = CountVectorizer(stop_words="english", max_features=20, min_df=5)
    X = cv.fit_transform(frame.clean_text)
    counts = np.asarray(X.sum(axis=0)).ravel()
    freq = pd.DataFrame({"term": cv.get_feature_names_out(), "count": counts}).sort_values("count", ascending=False)
    freq.to_csv(OUT / "top_word_frequencies.csv", index=False)
    top_freq = freq.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top_freq["term"], top_freq["count"])
    plt.title("Most frequent content words in 20 Newsgroups"); plt.xlabel("Count"); plt.tight_layout()
    plt.savefig(OUT / "word_frequency.png", dpi=180); plt.close()
    plt.figure(figsize=(12, 6))
    for term, group in trends.groupby("term"):
        plt.plot(group.year, group.relative_frequency, label=term)
    plt.title("Language and technology terms in Google Books (1990–2019)")
    plt.xlabel("Year"); plt.ylabel("Relative frequency (smoothed)"); plt.legend(ncol=4, fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "language_trends.png", dpi=180); plt.close()
    topics = pd.read_csv(OUT / "nmf_topics.csv")
    plt.figure(figsize=(10, 5)); plt.bar(topics.topic.astype(str), topics.documents_dominant)
    plt.title("Document distribution across NMF topics"); plt.xlabel("Topic"); plt.ylabel("Documents")
    plt.tight_layout(); plt.savefig(OUT / "topic_distribution.png", dpi=180); plt.close()


def main():
    print("[1/4] Downloading and preprocessing 20 Newsgroups...")
    frame = load_newsgroups()
    print(f"      Documents after cleaning: {len(frame)}")
    print("[2/4] Running LDA, NMF and K-means...")
    metrics = topic_models(frame)
    print("[3/4] Retrieving Google Books Ngram time series...")
    trends = fetch_ngram_trends()
    print("[4/4] Creating charts and summary...")
    make_charts(frame, trends)
    summary = {**metrics, "trend_terms": TERMS, "trend_years": "1990-2019",
               "dataset": "20 Newsgroups (all split; headers/footers/quotes removed) + Google Books Ngram 2019 English corpus",
               "random_seed": SEED}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nCompleted. Results saved in: {OUT}")

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        print("Check internet access, then run again. See README for troubleshooting.", file=sys.stderr)
        raise
