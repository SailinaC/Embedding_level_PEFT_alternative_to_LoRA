import os
import re
import random
import sqlite3
from pathlib import Path

import numpy as np
import torch
import ir_datasets

from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

SEED = 42
MODEL_NAME = "bert-large-uncased"
DOC_DB_PATH = "msmarco_passage_docs.sqlite"

OUTPUT_DIR = Path("outputs/full_monobert")
LATEST_DIR = OUTPUT_DIR / "latest"
FINAL_DIR = OUTPUT_DIR / "final"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LATEST_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

NUM_TRAIN_STEPS = 100000
NUM_WARMUP_STEPS = 10000
SAVE_EVERY = 5000
LOG_EVERY = 100

BATCH_SIZE = 8
ACCUMULATION_STEPS = 16
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 3e-6
WEIGHT_DECAY = 0.01

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, flush=True)

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

def save_checkpoint(path, step, model, optimizer, scheduler, scaler):
    path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_name": MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": {
            "num_train_steps": NUM_TRAIN_STEPS,
            "num_warmup_steps": NUM_WARMUP_STEPS,
            "batch_size": BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }

    temp_path = path / "checkpoint.tmp.pt"
    final_path = path / "checkpoint.pt"

    torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=False)
    os.replace(temp_path, final_path)

    print(f"Saved checkpoint at step {step}: {final_path}", flush=True)

def load_checkpoint(model, optimizer, scheduler, scaler):
    checkpoint_path = LATEST_DIR / "checkpoint.pt"

    if not checkpoint_path.exists():
        return 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    step = int(checkpoint["step"])

    print("Resumed from:", checkpoint_path, flush=True)
    print("Start step:", step, flush=True)

    return step

train_dataset = ir_datasets.load("msmarco-passage/train")
triples_dataset = ir_datasets.load("msmarco-passage/train/triples-small")

queries = {}

for query in train_dataset.queries_iter():
    queries[str(query.query_id)] = query.text

print("Loaded queries:", len(queries), flush=True)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

model.gradient_checkpointing_enable()
model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print("Trainable parameters:", trainable_params, flush=True)
print("Total model parameters:", total_params, flush=True)
print("Trainable percentage:", 100.0 * trainable_params / total_params, flush=True)
print("Batch size:", BATCH_SIZE, flush=True)
print("Accumulation steps:", ACCUMULATION_STEPS, flush=True)
print("Effective batch size:", BATCH_SIZE * ACCUMULATION_STEPS, flush=True)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=NUM_WARMUP_STEPS,
    num_training_steps=NUM_TRAIN_STEPS
)

scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

start_step = load_checkpoint(model, optimizer, scheduler, scaler)

def triple_iterator():
    while True:
        for triple in triples_dataset.docpairs_iter():
            yield triple

def next_pair(iterator):
    while True:
        triple = next(iterator)
        qid = str(triple.query_id)

        if qid not in queries:
            continue

        try:
            query_text = queries[qid]
            pos_doc_text = get_doc_text(triple.doc_id_a)
            neg_doc_text = get_doc_text(triple.doc_id_b)
            return (query_text, pos_doc_text, 1), (query_text, neg_doc_text, 0)
        except KeyError:
            continue

def make_batch(iterator, batch_size):
    batch = []

    while len(batch) < batch_size:
        pos_example, neg_example = next_pair(iterator)
        batch.append(pos_example)

        if len(batch) < batch_size:
            batch.append(neg_example)

    random.shuffle(batch)
    return batch

model.train()
iterator = triple_iterator()

running_loss = 0.0
running_count = 0

optimizer.zero_grad(set_to_none=True)

total_micro_steps = (NUM_TRAIN_STEPS - start_step) * ACCUMULATION_STEPS

for micro_step in range(total_micro_steps):
    batch_examples = make_batch(iterator, BATCH_SIZE)

    queries_batch = [x[0] for x in batch_examples]
    docs_batch = [x[1] for x in batch_examples]
    labels = torch.tensor([x[2] for x in batch_examples], dtype=torch.long, device=device)

    encoded = tokenizer(
        queries_batch,
        docs_batch,
        padding=True,
        truncation="only_second",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )

    encoded = {key: value.to(device) for key, value in encoded.items()}

    with autocast("cuda", enabled=torch.cuda.is_available()):
        outputs = model(**encoded, labels=labels)
        loss = outputs.loss / ACCUMULATION_STEPS

    scaler.scale(loss).backward()

    if (micro_step + 1) % ACCUMULATION_STEPS == 0:
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step = start_step + ((micro_step + 1) // ACCUMULATION_STEPS)

        running_loss += loss.item() * ACCUMULATION_STEPS
        running_count += 1

        if step % LOG_EVERY == 0:
            avg_loss = running_loss / running_count
            print(f"Step {step}/{NUM_TRAIN_STEPS} | loss = {avg_loss:.4f}", flush=True)
            running_loss = 0.0
            running_count = 0

        if step % SAVE_EVERY == 0:
            save_checkpoint(LATEST_DIR, step, model, optimizer, scheduler, scaler)

save_checkpoint(LATEST_DIR, NUM_TRAIN_STEPS, model, optimizer, scheduler, scaler)
save_checkpoint(FINAL_DIR, NUM_TRAIN_STEPS, model, optimizer, scheduler, scaler)

model.save_pretrained(FINAL_DIR / "hf_model")
tokenizer.save_pretrained(FINAL_DIR / "hf_model")

print("Finished full MonoBERT training", flush=True)
print("Saved final checkpoint:", FINAL_DIR / "checkpoint.pt", flush=True)
print("Saved HuggingFace model:", FINAL_DIR / "hf_model", flush=True)

doc_conn.close()
