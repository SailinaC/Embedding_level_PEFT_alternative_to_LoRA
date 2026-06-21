import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import ir_datasets
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_NAME = "bert-large-uncased"
CHECKPOINT_PATH = "outputs/approach6_lora_top3_attention/final/trainable_checkpoint.pt"
DOC_DB_PATH = "msmarco_passage_docs.sqlite"
HARD_CANDIDATE_FILE = "hard_bm25_top1000.tsv"

OUTPUT_DIR = Path("eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_JSONL = OUTPUT_DIR / "lora_top3_attention_results.jsonl"
RESULTS_CSV = OUTPUT_DIR / "lora_top3_attention_results.csv"

TOP_K = 1000
BATCH_SIZE = 64
MAX_LENGTH = 128

DATASETS = [
    "msmarco-passage/trec-dl-2019/judged",
    "msmarco-passage/trec-dl-2020/judged",
    "msmarco-passage/dev/small",
    "msmarco-passage/trec-dl-hard",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

doc_conn = sqlite3.connect(DOC_DB_PATH)
doc_cursor = doc_conn.cursor()

doc_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
DOC_TABLE = doc_cursor.fetchone()[0]

doc_cursor.execute(f"PRAGMA table_info({DOC_TABLE})")
DOC_COLUMNS = [row[1] for row in doc_cursor.fetchall()]

if "doc_id" in DOC_COLUMNS:
    DOC_ID_COLUMN = "doc_id"
elif "id" in DOC_COLUMNS:
    DOC_ID_COLUMN = "id"
else:
    DOC_ID_COLUMN = DOC_COLUMNS[0]

if "text" in DOC_COLUMNS:
    DOC_TEXT_COLUMN = "text"
elif "doc_text" in DOC_COLUMNS:
    DOC_TEXT_COLUMN = "doc_text"
else:
    DOC_TEXT_COLUMN = DOC_COLUMNS[1]

print("Document table:", DOC_TABLE, flush=True)
print("Document id column:", DOC_ID_COLUMN, flush=True)
print("Document text column:", DOC_TEXT_COLUMN, flush=True)

class LoRALinear(nn.Module):
    def __init__(self, base_layer, r, alpha, dropout):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = self.dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        return base_output + lora_output * self.scaling

def get_module_by_name(model, module_name):
    module = model

    for part in module_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    return module

def set_module_by_name(model, module_name, new_module):
    parts = module_name.split(".")
    parent = model

    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    setattr(parent, parts[-1], new_module)

def apply_lora_top3(model, target_modules, r, alpha, dropout):
    for module_name in target_modules:
        base_layer = get_module_by_name(model, module_name)

        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"Target is not nn.Linear: {module_name}")

        set_module_by_name(
            model,
            module_name,
            LoRALinear(base_layer, r, alpha, dropout)
        )

def load_trainable_state_dict(model, state):
    named_params = dict(model.named_parameters())

    for name, value in state.items():
        if name in named_params:
            named_params[name].data.copy_(value.to(named_params[name].device))

def normalize_id(value):
    value = str(value)

    if value.startswith("D") and value[1:].isdigit():
        return value[1:]

    return value

def get_doc_text(doc_id):
    doc_id = normalize_id(doc_id)
    doc_cursor.execute(
        f"SELECT {DOC_TEXT_COLUMN} FROM {DOC_TABLE} WHERE {DOC_ID_COLUMN} = ?",
        (doc_id,)
    )
    row = doc_cursor.fetchone()

    if row is None:
        raise KeyError(f"Missing doc_id: {doc_id}")

    return row[0]

def load_queries(dataset_name):
    dataset = ir_datasets.load(dataset_name)
    queries = {}

    for query in dataset.queries_iter():
        queries[str(query.query_id)] = query.text

    return queries

def load_qrels(dataset_name):
    dataset = ir_datasets.load(dataset_name)
    qrels = defaultdict(dict)

    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        docid = normalize_id(qrel.doc_id)
        relevance = int(qrel.relevance)
        qrels[qid][docid] = relevance

    return dict(qrels)

def load_ir_dataset_candidates(dataset_name, top_k):
    dataset = ir_datasets.load(dataset_name)
    candidates = defaultdict(list)
    seen = defaultdict(set)
    order = 0

    for scored_doc in dataset.scoreddocs_iter():
        qid = str(scored_doc.query_id)
        docid = normalize_id(scored_doc.doc_id)

        if docid in seen[qid]:
            continue

        score = getattr(scored_doc, "score", None)
        rank = getattr(scored_doc, "rank", None)

        if score is None:
            score = -rank if rank is not None else -order

        candidates[qid].append((docid, float(score), order))
        seen[qid].add(docid)
        order += 1

    final_candidates = {}

    for qid, docs in candidates.items():
        docs = sorted(docs, key=lambda x: (-x[1], x[2]))
        final_candidates[qid] = [docid for docid, score, idx in docs[:top_k]]

    return final_candidates

def load_hard_candidates(path, top_k):
    candidates = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        f.readline()

        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            qid = str(parts[0])
            docid = normalize_id(parts[1])
            rank = int(parts[2])
            score = float(parts[3])

            candidates[qid].append((docid, rank, score))

    final_candidates = {}

    for qid, docs in candidates.items():
        docs = sorted(docs, key=lambda x: (x[1], -x[2]))
        final_candidates[qid] = [docid for docid, rank, score in docs[:top_k]]

    return final_candidates

def score_pairs(model, tokenizer, query_texts, doc_texts):
    scores = []

    model.eval()

    with torch.no_grad():
        for start in range(0, len(query_texts), BATCH_SIZE):
            batch_queries = query_texts[start:start + BATCH_SIZE]
            batch_docs = doc_texts[start:start + BATCH_SIZE]

            encoded = tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation="only_second",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            logits = outputs.logits
            batch_scores = logits[:, 1]

            scores.extend(batch_scores.detach().cpu().tolist())

    return scores

def rerank_dataset(model, tokenizer, dataset_name):
    queries = load_queries(dataset_name)
    qrels = load_qrels(dataset_name)

    if dataset_name == "msmarco-passage/trec-dl-hard":
        candidates = load_hard_candidates(HARD_CANDIDATE_FILE, TOP_K)
    else:
        candidates = load_ir_dataset_candidates(dataset_name, TOP_K)

    rankings = {}

    qids = [qid for qid in qrels.keys() if qid in candidates and qid in queries]

    print("Queries:", len(queries), flush=True)
    print("Queries with qrels:", len(qrels), flush=True)
    print("Queries with candidates:", len(qids), flush=True)

    for idx, qid in enumerate(qids, start=1):
        docids = candidates[qid]
        query_text = queries[qid]

        query_texts = []
        doc_texts = []

        for docid in docids:
            query_texts.append(query_text)
            doc_texts.append(get_doc_text(docid))

        scores = score_pairs(model, tokenizer, query_texts, doc_texts)
        ranked = sorted(zip(docids, scores), key=lambda x: x[1], reverse=True)
        rankings[qid] = [docid for docid, score in ranked]

        if idx % 10 == 0 or idx == len(qids):
            print(f"Evaluated queries: {idx} / {len(qids)}", flush=True)

    return rankings, qrels

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

    for qid, query_qrels in qrels.items():
        if qid not in rankings:
            continue

        ranking = rankings[qid]

        ndcg_scores.append(ndcg_at_k(ranking, query_qrels, 10))
        mrr_scores.append(mrr_at_k(ranking, query_qrels, 10))
        map_scores.append(average_precision_at_k(ranking, query_qrels, 1000))
        recall_scores.append(recall_at_k(ranking, query_qrels, 100))

    if len(ndcg_scores) == 0:
        raise RuntimeError("No queries were evaluated")

    return {
        "NDCG@10": sum(ndcg_scores) / len(ndcg_scores),
        "MRR@10": sum(mrr_scores) / len(mrr_scores),
        "MAP@1000": sum(map_scores) / len(map_scores),
        "Recall@100": sum(recall_scores) / len(recall_scores),
        "evaluated_queries": len(ndcg_scores),
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
    print("Using device:", device, flush=True)
    print("Checkpoint:", CHECKPOINT_PATH, flush=True)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    config = checkpoint["config"]

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    for param in model.parameters():
        param.requires_grad = False

    apply_lora_top3(
        model,
        config["target_modules"],
        config["lora_r"],
        config["lora_alpha"],
        config["lora_dropout"]
    )

    load_trainable_state_dict(model, checkpoint["lora_state_dict"])
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])

    model.to(device)
    model.eval()

    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    results = []

    for dataset_name in DATASETS:
        print("Evaluating:", dataset_name, flush=True)

        rankings, qrels = rerank_dataset(model, tokenizer, dataset_name)
        metrics = evaluate_rankings(rankings, qrels)

        result = {
            "approach": "LoRA top-3 attention",
            "dataset": dataset_name,
            "top_k": TOP_K,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
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
