import os
import re
import gzip
from pathlib import Path

import ir_datasets
import pandas as pd
import pyterrier as pt

INDEX_PATH = "/home/s.chaudhary-thesis/var/pyterrier_msmarco_passage_index"
OUTPUT_PATH = "hard_bm25_top1000.tsv"
TOP_K = 1000

def normalize_id(x):
    x = str(x)
    match = re.search(r"\d+", x)
    if match is None:
        return x
    return match.group(0)

def clean_query(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def find_collection_file():
    roots = [
        Path.home() / ".ir_datasets",
        Path.home() / "ir_datasets",
        Path.cwd()
    ]

    names = [
        "collection.tsv",
        "collection.tsv.gz",
        "msmarco-docs.tsv",
        "msmarco-docs.tsv.gz"
    ]

    for root in roots:
        if not root.exists():
            continue

        for name in names:
            matches = list(root.rglob(name))
            if matches:
                return matches[0]

    raise FileNotFoundError("Could not find collection.tsv or collection.tsv.gz")

def open_text_file(path):
    path = Path(path)

    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")

    return open(path, "r", encoding="utf-8", errors="replace")

def iter_docs(collection_file):
    with open_text_file(collection_file) as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t", 1)

            if len(parts) != 2:
                continue

            yield {
                "docno": normalize_id(parts[0]),
                "text": parts[1]
            }

            if (i + 1) % 500000 == 0:
                print("Indexed input docs:", i + 1)

def build_index():
    collection_file = find_collection_file()
    print("Collection file:", collection_file)

    properties_path = os.path.join(INDEX_PATH, "data.properties")

    if os.path.exists(properties_path):
        print("Index already exists:", INDEX_PATH)
        return pt.IndexRef.of(properties_path)

    os.makedirs(INDEX_PATH, exist_ok=True)

    indexer = pt.IterDictIndexer(
        INDEX_PATH,
        meta={"docno": 32},
        fields=["text"],
        overwrite=True
    )

    index_ref = indexer.index(iter_docs(collection_file))
    print("Index built:", index_ref)

    return index_ref

def load_hard_topics():
    dataset = ir_datasets.load("msmarco-passage/trec-dl-hard")
    rows = []

    for query in dataset.queries_iter():
        rows.append({
            "qid": str(query.query_id),
            "query": clean_query(query.text)
        })

    return pd.DataFrame(rows)

if not pt.java.started():
    pt.java.init()

index_ref = build_index()
topics = load_hard_topics()

print("Hard queries:", len(topics))

bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", num_results=TOP_K)
results = bm25.transform(topics)

results = results[["qid", "docno", "rank", "score"]]
results.to_csv(OUTPUT_PATH, sep="\t", index=False)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(results))
print(results.head())
