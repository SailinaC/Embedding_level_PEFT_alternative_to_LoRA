import json
import math
from collections import defaultdict
from pathlib import Path

import ir_datasets

OUTPUT_DIR = Path("eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_JSONL = OUTPUT_DIR / "bm25_results.jsonl"
RESULTS_CSV = OUTPUT_DIR / "bm25_results.csv"

TOP_K = 1000
HARD_CANDIDATE_FILE = "hard_bm25_top1000.tsv"

DATASETS = [
    "msmarco-passage/trec-dl-2019/judged",
    "msmarco-passage/trec-dl-2020/judged",
    "msmarco-passage/dev/small",
    "msmarco-passage/trec-dl-hard",
]

def load_qrels(dataset_name):
    dataset = ir_datasets.load(dataset_name)
    qrels = defaultdict(dict)

    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        docid = str(qrel.doc_id)
        relevance = int(qrel.relevance)
        qrels[qid][docid] = relevance

    return dict(qrels)

def load_ir_dataset_candidates(dataset_name, top_k):
    dataset = ir_datasets.load(dataset_name)
    rankings = defaultdict(list)
    seen = defaultdict(set)
    order = 0

    for scored_doc in dataset.scoreddocs_iter():
        qid = str(scored_doc.query_id)
        docid = str(scored_doc.doc_id)

        if docid in seen[qid]:
            continue

        score = getattr(scored_doc, "score", None)
        rank = getattr(scored_doc, "rank", None)

        if score is None:
            score = -rank if rank is not None else -order

        rankings[qid].append((docid, float(score), order))
        seen[qid].add(docid)
        order += 1

    final_rankings = {}

    for qid, docs in rankings.items():
        docs = sorted(docs, key=lambda x: (-x[1], x[2]))
        final_rankings[qid] = [docid for docid, score, idx in docs[:top_k]]

    return final_rankings

def load_hard_candidates(path, top_k):
    rankings = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()

        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            qid = str(parts[0])
            docid = str(parts[1])
            rank = int(parts[2])
            score = float(parts[3])

            rankings[qid].append((docid, rank, score))

    final_rankings = {}

    for qid, docs in rankings.items():
        docs = sorted(docs, key=lambda x: (x[1], -x[2]))
        final_rankings[qid] = [docid for docid, rank, score in docs[:top_k]]

    return final_rankings

def dcg_at_k(relevances, k):
    total = 0.0

    for i, rel in enumerate(relevances[:k], start=1):
        total += (2.0 ** rel - 1.0) / math.log2(i + 1)

    return total

def ndcg_at_k(ranking, qrels, k):
    gains = [qrels.get(docid, 0) for docid in ranking[:k]]
    ideal = sorted(qrels.values(), reverse=True)

    ideal_dcg = dcg_at_k(ideal, k)

    if ideal_dcg == 0.0:
        return 0.0

    return dcg_at_k(gains, k) / ideal_dcg

def mrr_at_k(ranking, qrels, k):
    for i, docid in enumerate(ranking[:k], start=1):
        if qrels.get(docid, 0) > 0:
            return 1.0 / i

    return 0.0

def average_precision_at_k(ranking, qrels, k):
    relevant_total = sum(1 for rel in qrels.values() if rel > 0)

    if relevant_total == 0:
        return 0.0

    found = 0
    total = 0.0

    for i, docid in enumerate(ranking[:k], start=1):
        if qrels.get(docid, 0) > 0:
            found += 1
            total += found / i

    return total / relevant_total

def recall_at_k(ranking, qrels, k):
    relevant_docs = {docid for docid, rel in qrels.items() if rel > 0}

    if len(relevant_docs) == 0:
        return 0.0

    retrieved = set(ranking[:k])
    return len(relevant_docs & retrieved) / len(relevant_docs)

def evaluate_rankings(rankings, qrels):
    ndcg_scores = []
    mrr_scores = []
    map_scores = []
    recall_scores = []

    evaluated = 0

    for qid, query_qrels in qrels.items():
        if qid not in rankings:
            continue

        ranking = rankings[qid]

        ndcg_scores.append(ndcg_at_k(ranking, query_qrels, 10))
        mrr_scores.append(mrr_at_k(ranking, query_qrels, 10))
        map_scores.append(average_precision_at_k(ranking, query_qrels, 1000))
        recall_scores.append(recall_at_k(ranking, query_qrels, 100))

        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No queries were evaluated")

    return {
        "NDCG@10": sum(ndcg_scores) / len(ndcg_scores),
        "MRR@10": sum(mrr_scores) / len(mrr_scores),
        "MAP@1000": sum(map_scores) / len(map_scores),
        "Recall@100": sum(recall_scores) / len(recall_scores),
        "evaluated_queries": evaluated,
    }

def append_result(result):
    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

def write_csv(results):
    with open(RESULTS_CSV, "w", encoding="utf-8") as f:
        f.write("approach,dataset,NDCG@10,MRR@10,MAP@1000,Recall@100,evaluated_queries\n")

        for result in results:
            metrics = result["metrics"]
            f.write(
                f"{result['approach']},"
                f"{result['dataset']},"
                f"{metrics['NDCG@10']},"
                f"{metrics['MRR@10']},"
                f"{metrics['MAP@1000']},"
                f"{metrics['Recall@100']},"
                f"{metrics['evaluated_queries']}\n"
            )

def main():
    results = []

    for dataset_name in DATASETS:
        print("Evaluating:", dataset_name, flush=True)

        qrels = load_qrels(dataset_name)

        if dataset_name == "msmarco-passage/trec-dl-hard":
            rankings = load_hard_candidates(HARD_CANDIDATE_FILE, TOP_K)
        else:
            rankings = load_ir_dataset_candidates(dataset_name, TOP_K)

        print("Queries with qrels:", len(qrels), flush=True)
        print("Queries with rankings:", len(rankings), flush=True)

        metrics = evaluate_rankings(rankings, qrels)

        result = {
            "approach": "BM25",
            "dataset": dataset_name,
            "top_k": TOP_K,
            "metrics": metrics,
        }

        print(json.dumps(result, indent=2), flush=True)

        append_result(result)
        results.append(result)

    write_csv(results)

    print("Saved:", RESULTS_JSONL, flush=True)
    print("Saved:", RESULTS_CSV, flush=True)

if __name__ == "__main__":
    main()
