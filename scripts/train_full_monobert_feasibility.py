import random
import sqlite3
import time

import numpy as np
import torch
import ir_datasets

from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

SEED = 42
MODEL_NAME = "bert-large-uncased"
DOC_DB_PATH = "msmarco_passage_docs.sqlite"

NUM_TRAIN_STEPS = 1000
NUM_WARMUP_STEPS = 100
LOG_EVERY = 50

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

start_time = time.time()

total_micro_steps = NUM_TRAIN_STEPS * ACCUMULATION_STEPS

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

        step = (micro_step + 1) // ACCUMULATION_STEPS

        running_loss += loss.item() * ACCUMULATION_STEPS
        running_count += 1

        if step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / running_count
            seconds_per_step = elapsed / step
            estimated_100k_hours = (seconds_per_step * 100000) / 3600.0

            if torch.cuda.is_available():
                memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            else:
                memory_gb = 0.0

            print(
                f"Step {step}/{NUM_TRAIN_STEPS} | loss = {avg_loss:.4f} | "
                f"sec/step = {seconds_per_step:.4f} | "
                f"estimated_100k_hours = {estimated_100k_hours:.2f} | "
                f"max_gpu_memory_gb = {memory_gb:.2f}",
                flush=True
            )

            running_loss = 0.0
            running_count = 0

elapsed = time.time() - start_time
seconds_per_step = elapsed / NUM_TRAIN_STEPS
estimated_100k_hours = (seconds_per_step * 100000) / 3600.0

print("Finished feasibility test", flush=True)
print("Elapsed seconds:", elapsed, flush=True)
print("Seconds per step:", seconds_per_step, flush=True)
print("Estimated hours for 100k steps:", estimated_100k_hours, flush=True)

if torch.cuda.is_available():
    print("Max GPU memory GB:", torch.cuda.max_memory_allocated() / (1024 ** 3), flush=True)

doc_conn.close()
