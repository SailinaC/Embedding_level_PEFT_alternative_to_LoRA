import os
import gzip
import random
import re
import sqlite3
from pathlib import Path

import numpy as np
import torch
import ir_datasets

from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

SEED = 42
MODEL_NAME = "bert-large-uncased"
NUM_TRAIN_STEPS = 100000
NUM_WARMUP_STEPS = 10000
BATCH_SIZE = 16
ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 1e-4
OUTPUT_DIR = "outputs/approach1_cls_sep_only"
CHECKPOINT_EVERY = 10000
LOG_EVERY = 100
DOC_DB_PATH = "msmarco_passage_docs.sqlite"
RESUME_FROM = "outputs/approach1_cls_sep_only/checkpoint-20000"
USE_AMP = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_id(x):
    x = str(x)
    match = re.search(r"\d+", x)
    if match is None:
        return x
    return match.group(0)

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

def build_doc_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, text TEXT)")
    cursor.execute("SELECT COUNT(*) FROM docs")
    count = cursor.fetchone()[0]

    if count >= 8841823:
        print("Document database already exists with rows:", count)
        conn.close()
        return

    collection_file = find_collection_file()
    print("Using collection file:", collection_file)
    print("Building or repairing document database on disk")

    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA synchronous = NORMAL")

    batch = []

    with open_text_file(collection_file) as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t", 1)

            if len(parts) != 2:
                continue

            doc_id = normalize_id(parts[0])
            text = parts[1]
            batch.append((doc_id, text))

            if len(batch) >= 20000:
                cursor.executemany("INSERT OR REPLACE INTO docs VALUES (?, ?)", batch)
                conn.commit()
                batch = []

            if (i + 1) % 500000 == 0:
                cursor.execute("SELECT COUNT(*) FROM docs")
                count = cursor.fetchone()[0]
                print("Scanned:", i + 1, "Rows in db:", count)

    if batch:
        cursor.executemany("INSERT OR REPLACE INTO docs VALUES (?, ?)", batch)
        conn.commit()

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_id ON docs(id)")
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM docs")
    count = cursor.fetchone()[0]
    print("Final document rows:", count)

    conn.close()

build_doc_database(DOC_DB_PATH)

doc_conn = sqlite3.connect(DOC_DB_PATH)
doc_cursor = doc_conn.cursor()

def get_doc_text(doc_id):
    doc_id = normalize_id(doc_id)
    doc_cursor.execute("SELECT text FROM docs WHERE id = ?", (doc_id,))
    row = doc_cursor.fetchone()

    if row is None:
        raise KeyError(f"Missing doc_id: {doc_id}")

    return row[0]

train_dataset = ir_datasets.load("msmarco-passage/train")
queries = {str(q.query_id): q.text for q in train_dataset.queries_iter()}
print("Loaded queries:", len(queries))

triples_dataset = ir_datasets.load("msmarco-passage/train/triples-small")
triples_iter = triples_dataset.docpairs_iter()

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

cls_id = tokenizer.cls_token_id
sep_id = tokenizer.sep_token_id
selected_token_ids = [cls_id, sep_id]

print("[CLS] id:", cls_id)
print("[SEP] id:", sep_id)
print("Selected trainable token ids:", selected_token_ids)
print("Selected tokens:", tokenizer.convert_ids_to_tokens(selected_token_ids))

for param in model.parameters():
    param.requires_grad = False

embedding_weight = model.bert.embeddings.word_embeddings.weight
embedding_weight.requires_grad = True

def mask_embedding_gradients(grad):
    mask = torch.zeros_like(grad)
    for token_id in selected_token_ids:
        mask[token_id] = 1.0
    return grad * mask

embedding_weight.register_hook(mask_embedding_gradients)

actual_trainable_params = len(selected_token_ids) * model.config.hidden_size
optimizer_seen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Effective trainable embedding params:", actual_trainable_params)
print("Optimizer-visible params:", optimizer_seen_params)

optimizer = AdamW(
    [embedding_weight],
    lr=LEARNING_RATE,
    weight_decay=0.0
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=NUM_WARMUP_STEPS,
    num_training_steps=NUM_TRAIN_STEPS
)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and USE_AMP))

start_step = 0

def save_checkpoint(path, step):
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_name": MODEL_NAME,
        "selected_token_ids": selected_token_ids,
        "embedding_rows": {
            str(token_id): embedding_weight[token_id].detach().cpu()
            for token_id in selected_token_ids
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict()
    }

    torch.save(checkpoint, os.path.join(path, "trainable_checkpoint.pt"))

def load_checkpoint(path):
    checkpoint = torch.load(
        os.path.join(path, "trainable_checkpoint.pt"),
        map_location=device
    )

    with torch.no_grad():
        for token_id_string, value in checkpoint["embedding_rows"].items():
            embedding_weight[int(token_id_string)].copy_(value.to(device))

    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint["step"]

if RESUME_FROM is not None:
    start_step = load_checkpoint(RESUME_FROM)
    print("Resumed from step:", start_step)

initial_embedding = embedding_weight.detach().clone()

def next_pair_from_triple_iterator(iterator):
    triple = next(iterator)
    query_text = queries[str(triple.query_id)]
    pos_doc_text = get_doc_text(triple.doc_id_a)
    neg_doc_text = get_doc_text(triple.doc_id_b)
    return (query_text, pos_doc_text, 1), (query_text, neg_doc_text, 0)

def make_batch(iterator, batch_size):
    batch = []

    while len(batch) < batch_size:
        pos_example, neg_example = next_pair_from_triple_iterator(iterator)
        batch.append(pos_example)

        if len(batch) < batch_size:
            batch.append(neg_example)

    return batch

if start_step > 0:
    examples_to_skip = start_step * ACCUMULATION_STEPS * BATCH_SIZE
    triples_to_skip = examples_to_skip // 2
    print("Skipping triples:", triples_to_skip)

    for _ in range(triples_to_skip):
        next(triples_iter)

model.train()
optimizer.zero_grad(set_to_none=True)

running_loss = 0.0
running_count = 0
opt_step = start_step
total_micro_steps = (NUM_TRAIN_STEPS - start_step) * ACCUMULATION_STEPS

for micro_step in range(total_micro_steps):
    batch_examples = make_batch(triples_iter, BATCH_SIZE)

    queries_batch = [x[0] for x in batch_examples]
    docs_batch = [x[1] for x in batch_examples]
    labels = torch.tensor([x[2] for x in batch_examples], dtype=torch.long, device=device)

    encoded = tokenizer(
        queries_batch,
        docs_batch,
        padding=True,
        truncation="only_second",
        max_length=MAX_SEQ_LENGTH,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device, non_blocking=True)
    attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
    token_type_ids = encoded["token_type_ids"].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and USE_AMP), dtype=torch.float16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

        loss = outputs.loss / ACCUMULATION_STEPS

    scaler.scale(loss).backward()

    running_loss += outputs.loss.detach().float().item()
    running_count += 1

    if (micro_step + 1) % ACCUMULATION_STEPS == 0:
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        opt_step += 1

        if opt_step % LOG_EVERY == 0:
            print(f"Step {opt_step}/{NUM_TRAIN_STEPS} | loss = {running_loss / running_count:.4f}")
            running_loss = 0.0
            running_count = 0

        if opt_step % CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint-{opt_step}")
            save_checkpoint(checkpoint_path, opt_step)
            print("Saved checkpoint:", checkpoint_path)

        if opt_step >= NUM_TRAIN_STEPS:
            break

final_path = os.path.join(OUTPUT_DIR, "final")
save_checkpoint(final_path, opt_step)

changed_rows = torch.nonzero(
    torch.norm(embedding_weight.detach() - initial_embedding, dim=1) > 1e-8
).squeeze(-1).cpu().tolist()

print("Saved final checkpoint:", final_path)
print("Changed embedding rows:", changed_rows)
print("Changed tokens:", tokenizer.convert_ids_to_tokens(changed_rows))

for token_id in selected_token_ids:
    diff = torch.norm(embedding_weight[token_id].detach() - initial_embedding[token_id]).item()
    print(f"Embedding change for {tokenizer.convert_ids_to_tokens([token_id])[0]} id {token_id}: {diff:.8f}")

unexpected_changed = [x for x in changed_rows if x not in selected_token_ids]
print("Unexpected changed embedding rows:", unexpected_changed)

doc_conn.close()


