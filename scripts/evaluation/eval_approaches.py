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
        query_id = str(qrel.query_id)
        doc_id = normalize_id(qrel.doc_id)
        relevance = int(qrel.relevance)

        if relevance > 0:
            qrels[query_id][doc_id] = relevance

    return dict(qrels)

def load_candidates(dataset, top_k):
    candidates = defaultdict(list)

    for scored_doc in dataset.scoreddocs_iter():
        query_id = str(scored_doc.query_id)
        doc_id = normalize_id(scored_doc.doc_id)

        if len(candidates[query_id]) < top_k:
            candidates[query_id].append(doc_id)

    return dict(candidates)

def connect_doc_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    return conn, cursor

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
    ideal_gains = sorted(relevant_docs.values(), reverse=True)

    dcg = dcg_at_k(gains, k)
    idcg = dcg_at_k(ideal_gains, k)

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
    ndcg_scores = []
    mrr_scores = []
    map_scores = []
    recall_scores = []

    for query_id, relevant_docs in qrels.items():
        ranked_docs = results.get(query_id, [])

        ndcg_scores.append(ndcg_at_k(relevant_docs, ranked_docs, 10))
        mrr_scores.append(reciprocal_rank_at_k(relevant_docs, ranked_docs, 10))
        map_scores.append(average_precision(relevant_docs, ranked_docs, 1000))
        recall_scores.append(recall_at_k(relevant_docs, ranked_docs, 100))

    return {
        "NDCG@10": float(np.mean(ndcg_scores)),
        "MRR@10": float(np.mean(mrr_scores)),
        "MAP@1000": float(np.mean(map_scores)),
        "Recall@100": float(np.mean(recall_scores))
    }

def load_model(approach_name, device):
    set_seed(SEED)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    checkpoint_path = APPROACHES[approach_name]["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if APPROACHES[approach_name]["type"] == "embeddings":
        embedding_weight = model.bert.embeddings.word_embeddings.weight

        with torch.no_grad():
            for token_id_string, value in checkpoint["embedding_rows"].items():
                embedding_weight[int(token_id_string)].copy_(value)

    elif APPROACHES[approach_name]["type"] == "matrices":
        model_state = dict(model.named_parameters())

        with torch.no_grad():
            for name, value in checkpoint["trainable_state"].items():
                model_state[name].copy_(value)

    else:
        raise ValueError(f"Unknown approach type: {APPROACHES[approach_name]['type']}")

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

    scores = logits[:, 1].detach().float().cpu().tolist()
    return scores

def rerank_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Approach:", args.approach)
    print("Dataset:", args.dataset)

    dataset = ir_datasets.load(args.dataset)

    if not dataset.has_scoreddocs():
        raise RuntimeError(f"Dataset does not have scoreddocs: {args.dataset}")

    queries = load_queries(dataset)
    qrels = load_qrels(dataset)
    candidates = load_candidates(dataset, args.top_k)

    query_ids = sorted(qrels.keys())

    if args.max_queries is not None:
        query_ids = query_ids[:args.max_queries]

    print("Queries with qrels:", len(qrels))
    print("Queries to evaluate:", len(query_ids))
    print("Candidate top_k:", args.top_k)

    conn, cursor = connect_doc_db(DOC_DB_PATH)
    tokenizer, model = load_model(args.approach, device)

    results = {}

    for query_index, query_id in enumerate(query_ids, start=1):
        if query_id not in queries:
            continue

        doc_ids = candidates.get(query_id, [])

        if not doc_ids:
            continue

        query_text = queries[query_id]
        scored = []

        for start in range(0, len(doc_ids), args.batch_size):
            batch_doc_ids = doc_ids[start:start + args.batch_size]
            batch_queries = [query_text] * len(batch_doc_ids)
            batch_docs = [get_doc_text(cursor, doc_id) for doc_id in batch_doc_ids]

            batch_scores = score_batch(
                tokenizer,
                model,
                device,
                batch_queries,
                batch_docs,
                args.max_length
            )

            scored.extend(zip(batch_doc_ids, batch_scores))

        scored.sort(key=lambda x: x[1], reverse=True)
        results[query_id] = [doc_id for doc_id, score in scored]

        if query_index % 10 == 0:
            print("Evaluated queries:", query_index, "/", len(query_ids))

    conn.close()

    metrics = evaluate_metrics(
        {qid: qrels[qid] for qid in query_ids if qid in qrels},
        results
    )

    output = {
        "approach": args.approach,
        "dataset": args.dataset,
        "top_k": args.top_k,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_queries": args.max_queries,
        "metrics": metrics
    }

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(output) + "\n")

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=sorted(APPROACHES.keys()), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results")

    args = parser.parse_args()
    rerank_dataset(args)
