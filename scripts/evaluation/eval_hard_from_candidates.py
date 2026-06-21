import argparse
import json
import math
import os
import random
import re
import sqlite3
from collections import defaultdict

import numpy as np
import torch
import ir_datasets

from transformers import BertTokenizer, BertForSequenceClassification

SEED = 42
MODEL_NAME = "bert-large-uncased"
DOC_DB_PATH = "msmarco_passage_docs.sqlite"

APPROACHES = {
    "approach1": {
        "checkpoint": "outputs/approach1_cls_sep_only/final/trainable_checkpoint.pt",
        "type": "embeddings"
    },
    "approach2": {
        "checkpoint": "outputs/approach2_top6_embeddings/final/trainable_checkpoint.pt",
        "type": "embeddings"
    },
    "approach3": {
        "checkpoint": "outputs/approach3_top3_attention/final/trainable_checkpoint.pt",
        "type": "matrices"
    }
}

def normalize_id(x):
    x = str(x)
    match = re.search(r"\d+", x)
    if match is None:
        return x
    return match.group(0)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_queries(dataset):
    return {str(q.query_id): q.text for q in dataset.queries_iter()}

def load_qrels(dataset):
    qrels = defaultdict(dict)

    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        docid = normalize_id(qrel.doc_id)
        rel = int(qrel.relevance)

        if rel > 0:
            qrels[qid][docid] = rel

    return dict(qrels)

def load_candidates(path, top_k):
    candidates = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        header = next(f)
        cols = header.rstrip("\n").split("\t")
        qid_i = cols.index("qid")
        doc_i = cols.index("docno")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            qid = str(parts[qid_i])
            docid = normalize_id(parts[doc_i])

            if len(candidates[qid]) < top_k:
                candidates[qid].append(docid)

    return dict(candidates)

def get_doc_text(cursor, doc_id):
    doc_id = normalize_id(doc_id)
    cursor.execute("SELECT text FROM docs WHERE id = ?", (doc_id,))
    row = cursor.fetchone()

    if row is None:
        raise KeyError(f"Missing doc_id: {doc_id}")

    return row[0]

def average_precision(relevant_docs, ranked_docs, k):
    score = 0.0
    hits = 0

    for rank, doc_id in enumerate(ranked_docs[:k], start=1):
        if doc_id in relevant_docs:
            hits += 1
            score += hits / rank

    if not relevant_docs:
        return 0.0

    return score / min(len(relevant_docs), k)

def dcg_at_k(relevances, k):
    score = 0.0

    for i, rel in enumerate(relevances[:k], start=1):
        score += (2 ** rel - 1) / math.log2(i + 1)

    return score

def ndcg_at_k(relevant_docs, ranked_docs, k):
    gains = [relevant_docs.get(doc_id, 0) for doc_id in ranked_docs[:k]]
    ideal = sorted(relevant_docs.values(), reverse=True)

    dcg = dcg_at_k(gains, k)
    idcg = dcg_at_k(ideal, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg

def reciprocal_rank_at_k(relevant_docs, ranked_docs, k):
    for rank, doc_id in enumerate(ranked_docs[:k], start=1):
        if doc_id in relevant_docs:
            return 1.0 / rank

    return 0.0

def recall_at_k(relevant_docs, ranked_docs, k):
    if not relevant_docs:
        return 0.0

    hits = sum(1 for doc_id in ranked_docs[:k] if doc_id in relevant_docs)
    return hits / len(relevant_docs)

def evaluate_metrics(qrels, results):
    ndcg = []
    mrr = []
    map_scores = []
    recall = []

    for qid, relevant_docs in qrels.items():
        ranked_docs = results.get(qid, [])

        ndcg.append(ndcg_at_k(relevant_docs, ranked_docs, 10))
        mrr.append(reciprocal_rank_at_k(relevant_docs, ranked_docs, 10))
        map_scores.append(average_precision(relevant_docs, ranked_docs, 1000))
        recall.append(recall_at_k(relevant_docs, ranked_docs, 100))

    return {
        "NDCG@10": float(np.mean(ndcg)),
        "MRR@10": float(np.mean(mrr)),
        "MAP@1000": float(np.mean(map_scores)),
        "Recall@100": float(np.mean(recall))
    }

def load_model(approach, device):
    set_seed(SEED)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    checkpoint = torch.load(APPROACHES[approach]["checkpoint"], map_location=device)

    if APPROACHES[approach]["type"] == "embeddings":
        emb = model.bert.embeddings.word_embeddings.weight

        with torch.no_grad():
            for token_id, value in checkpoint["embedding_rows"].items():
                emb[int(token_id)].copy_(value)

    elif APPROACHES[approach]["type"] == "matrices":
        params = dict(model.named_parameters())

        with torch.no_grad():
            for name, value in checkpoint["trainable_state"].items():
                params[name].copy_(value)

    else:
        raise ValueError(f"Unknown approach type: {APPROACHES[approach]['type']}")

    model.to(device)
    model.eval()

    return tokenizer, model

def score_batch(tokenizer, model, device, queries, docs, max_length):
    encoded = tokenizer(
        queries,
        docs,
        padding=True,
        truncation="only_second",
        max_length=max_length,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = encoded["token_type_ids"].to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ).logits

    return logits[:, 1].detach().float().cpu().tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=sorted(APPROACHES.keys()), required=True)
    parser.add_argument("--candidate_file", default="hard_bm25_top1000.tsv")
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", default="eval_results")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ir_datasets.load("msmarco-passage/trec-dl-hard")

    print("Using device:", device)
    print("Approach:", args.approach)
    print("Dataset: msmarco-passage/trec-dl-hard")

    queries = load_queries(dataset)
    qrels = load_qrels(dataset)
    candidates = load_candidates(args.candidate_file, args.top_k)

    print("Queries:", len(queries))
    print("Queries with qrels:", len(qrels))
    print("Queries with candidates:", len(candidates))

    conn = sqlite3.connect(DOC_DB_PATH)
    cursor = conn.cursor()

    tokenizer, model = load_model(args.approach, device)

    results = {}
    qids = sorted(qrels.keys())

    for qi, qid in enumerate(qids, start=1):
        doc_ids = candidates.get(qid, [])
        query_text = queries[qid]
        scored = []

        for start in range(0, len(doc_ids), args.batch_size):
            batch_doc_ids = doc_ids[start:start + args.batch_size]
            batch_queries = [query_text] * len(batch_doc_ids)
            batch_docs = [get_doc_text(cursor, docid) for docid in batch_doc_ids]

            scores = score_batch(
                tokenizer,
                model,
                device,
                batch_queries,
                batch_docs,
                args.max_length
            )

            scored.extend(zip(batch_doc_ids, scores))

        scored.sort(key=lambda x: x[1], reverse=True)
        results[qid] = [docid for docid, score in scored]

        if qi % 10 == 0:
            print("Evaluated queries:", qi, "/", len(qids))

    conn.close()

    metrics = evaluate_metrics(qrels, results)

    output = {
        "approach": args.approach,
        "dataset": "msmarco-passage/trec-dl-hard",
        "candidate_file": args.candidate_file,
        "top_k": args.top_k,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "metrics": metrics
    }

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(output) + "\n")

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
