import argparse
import logging
import time
import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import get_linear_schedule_with_warmup
import ir_datasets

# Setup arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_CLS', action='store_true')
parser.add_argument('--train_SEP', action='store_true')
parser.add_argument('--name', type=str, default='CLS_SEP')
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument('--num_train_steps', type=int, default=100000)
parser.add_argument('--num_warmup_steps', type=int, default=10000)
parser.add_argument('--learning_rate', type=float, default=3e-6)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=16)
parser.add_argument('--max_seq_length', type=int, default=512)
args = parser.parse_args()

# Randomness and device
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# Data loading function
def iter_train_samples():
    print("Loading MS MARCO triples...")
    dataset = ir_datasets.load('msmarco-passage/train/triples-small')
    docstore = ir_datasets.load('msmarco-passage/train').docs_store()
    queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/train').queries_iter()}

    while True:
        for triple in dataset.docpairs_iter():
            query_text = queries[triple.query_id]
            pos_doc_text = docstore.get(triple.doc_id_a).text
            neg_doc_text = docstore.get(triple.doc_id_b).text
            yield query_text, pos_doc_text, 1
            yield query_text, neg_doc_text, 0

train_iter = iter(iter_train_samples())

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(args.model_name)
model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

print(f"CLS ID: {tokenizer.cls_token_id}, SEP ID: {tokenizer.sep_token_id}")

for param in model.parameters():
    param.requires_grad = True

# Gradient masking hook
def mask_gradients(grad):
    mask = torch.zeros_like(grad)
    if args.train_CLS:
        mask[101] = 1.0 # CLS token ID
    if args.train_SEP:
        mask[102] = 1.0 # SEP token ID
    return grad * mask

embedding_weight = model.bert.embeddings.word_embeddings.weight
embedding_weight.requires_grad = True
embedding_weight.register_hook(mask_gradients)

# Save initial values to check later
initial_cls = embedding_weight[101].clone().detach()
initial_sep = embedding_weight[102].clone().detach()
initial_rand = embedding_weight[1000].clone().detach()

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=args.num_warmup_steps, 
    num_training_steps=args.num_train_steps
)

# Training loop
total_passes = args.num_train_steps * args.accumulation_steps
model.train()
optimizer.zero_grad()
total_loss = 0.0
count = 0
opt_step = 0
start_time = time.time()

for k in range(total_passes):
    # Fetch batch
    q_batch, d_batch, l_batch = [], [], []
    for _ in range(args.batch_size):
        q, d, l = next(train_iter)
        q_batch.append(q)
        d_batch.append(d)
        l_batch.append(l)

    encoded = tokenizer(q_batch, d_batch, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors='pt')
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device)
    label_tensor = torch.tensor(l_batch).to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=label_tensor)
    loss = outputs.loss / args.accumulation_steps
    loss.backward()

    total_loss += outputs.loss.item()
    count += 1

    if (k + 1) % args.accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        opt_step += 1

        if opt_step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step {opt_step}/{args.num_train_steps} | Loss: {total_loss/count:.4f}")

        if opt_step % 10000 == 0:
            model.save_pretrained(f'data/checkpoint-{opt_step}')

        if opt_step >= args.num_train_steps:
            break

# Save and verify
model.save_pretrained(f'data/bert-final-{args.name}')
print("Model saved.")

final_cls = model.bert.embeddings.word_embeddings.weight[101].detach()
final_sep = model.bert.embeddings.word_embeddings.weight[102].detach()
final_rand = model.bert.embeddings.word_embeddings.weight[1000].detach()

print("\nVerification:")
print(f"CLS changed: {not torch.equal(initial_cls.cpu(), final_cls.cpu())}")
print(f"SEP changed: {not torch.equal(initial_sep.cpu(), final_sep.cpu())}")
print(f"Other changed: {not torch.equal(initial_rand.cpu(), final_rand.cpu())}")
