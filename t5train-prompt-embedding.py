import json
import ir_datasets
import pandas as pd
import pyterrier as pt
#pt.init()
import wandb
import ir_datasets
import logging

import numpy as np
from pyterrier.measures import *
from pyterrier_t5 import MonoT5ReRanker
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW, Adafactor
from random import Random
import itertools
from datasets import load_dataset
BATCH_SIZE = 8

from pyterrier_pisa import PisaIndex

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--negs', type=int, default=1)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--train_Query', type=bool, default = False)
parser.add_argument('--train_Document', type=bool, default = False)
parser.add_argument('--train_Relevan', type=bool, default = False)
parser.add_argument('--train_true_false', type=bool, default = False)
parser.add_argument('--train_eos', type=bool, default = False)
parser.add_argument('--train_colon', type=bool, default = False)
parser.add_argument('--name', type=str, default = 'QDR')


args = parser.parse_args()

rng = np.random.RandomState(args.seed)

#wandb.init(
#    project="t5-train",
#    config={
#      'model': 'monot5',
#      'desc': 'debug',
#      'negs': args.negs,
#      'seed': args.seed,
#    }
#)

import torch
torch.manual_seed(args.seed)
_logger = ir_datasets.log.easy()

#print('PROMPT EXAMPLE:')
#print(args.query_token+': Who is Andy Murray?' + ' '+args.doc_token+': ' + 'Andy Murray is a tennis player' + ' Relevant:')


logging.basicConfig(filename='./Sub_embedding_t5_all_embd.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )

logger = logging.getLogger('main')
logging.info(f'Loading dataset')

OUTPUTS = ['true', 'false']

print(args.train_Query)
if args.train_Query:
    print('yeeeeeeeeeeee')

def iter_train_samples():
  dataset_triple = load_dataset('irds/msmarco-passage_train_triples-small', 'docpairs', trust_remote_code=True)
  dataset = ir_datasets.load('msmarco-passage/train')
  docs = dataset.docs_store()
  queries = {q.query_id: q.text for q in dataset.queries_iter()}
  while True:
    for triple in dataset_triple:
      yield  'Query: ' + queries[triple['query_id']] + ' Document: ' + docs.get(triple['doc_id_a']).text + ' Relevant:', OUTPUTS[0]
      yield 'Query: ' + queries[triple['query_id']] + ' Document: ' + docs.get(triple['doc_id_b']).text + ' Relevant:', OUTPUTS[1]

train_iter = _logger.pbar(iter_train_samples(), desc='total train samples')

model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
print(model.state_dict()['shared.weight'].shape)

tokenizer = T5Tokenizer.from_pretrained("t5-base")
print(len(tokenizer))

#special_tokens_dict = {'additional_special_tokens': ['[newtok1]','[newtok2]','[newtok3]']}
#num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
#tokenizer.add_tokens(['[newtok1]','[newtok2]','[newtok3]'], special_tokens=True)
#print(len(tokenizer))
#model.resize_token_embeddings(len(tokenizer))

#print(tokenizer.encode('[newtok1]: Who is Andy Murray? [newtok2]: Andy Murray is a tennis player [newtok3]:'))

#print(tokenizer.encode('[newtok1]'))

#print(model.state_dict()['shared.weight'].shape)

#tokenizer.save_pretrained('data/Tokenizer_t5-base-new_special_tokens3')

for param in model.parameters():
    param.requires_grad = True

#for name, param in model.named_parameters():
#    if 'block.11.' in name: 
#        print(name)
#        param.requires_grad = True
        

#initial_embedding = embedding_weight.clone()

#import random

#n_list = []
#for i in range(7):
#    n = random.randint(1, 32128)
#    print(tokenizer.decode([n]))
#    n_list.append(n)
    

# Register hook to zero out gradients for all but new token IDs
def mask_gradients(grad):
    mask = torch.zeros_like(grad)
    if args.train_Query:
        mask[27569] = 1.0
        mask[3] = 1.0
    if args.train_Document:
        mask[11167] = 1.0
    if args.train_Relevan:
        mask[31484] = 1.0
    if args.train_true_false:
        mask[1176] = 1.0
        mask[6136] = 1.0
    if args.train_eos:
        mask[1] = 1.0
    if args.train_colon:
        mask[10] = 1.0
    
    #mask[tokenizer.encode(['[newtok1]'])[0]]=1.0
    #mask[tokenizer.encode(['[newtok2]'])[0]]=1.0
    #mask[tokenizer.encode(['[newtok3]'])[0]]=1.0


#if args.train_random:
#    for i in range(7):
#        mask[n_list[i]] = 1.0
    
    return grad * mask

embedding_weight = model.get_input_embeddings().weight
embedding_weight.requires_grad = True
#embedding_weight.register_hook(mask_gradients)

optimizer = Adafactor(model.parameters(), lr=3e-4, weight_decay=5e-5)

reranker = MonoT5ReRanker(verbose=False, batch_size=BATCH_SIZE)
reranker.REL = tokenizer.encode(OUTPUTS[0])[0]
reranker.NREL = tokenizer.encode(OUTPUTS[1])[0]
        
        #def build_validation_data():
        #  result = []
        #  dataset = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')
        #  docs = dataset.docs_store()
        #  queries = {q.query_id: q.text for q in dataset.queries_iter()}
        #  for qrel in _logger.pbar(ir_datasets.load('msmarco-passage/trec-dl-2019/judged').scoreddocs, desc='dev data'):
        #    if qrel.query_id in queries:
        #      result.append([qrel.query_id, queries[qrel.query_id], qrel.doc_id, docs.get(qrel.doc_id).text])
        #  return pd.DataFrame(result, columns=['qid', 'query', 'docno', 'text'])
        
        #valid_data = build_validation_data()
        #valid_qrels = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged').get_qrels()
        
epoch = 0

#max_ndcg = 0.

number_df = 640000*10 #*10 #39780800
print(number_df)

import time
start = time.time()
#while True:
with _logger.pbar_raw(desc=f'train {epoch}', total=number_df // BATCH_SIZE) as pbar:
    model.train()
    total_loss = 0
    count = 0
    for k in range(number_df // BATCH_SIZE):
      inp, out = [], []
      for i in range(BATCH_SIZE):
        i, o = next(train_iter)
        inp.append(i)
        out.append(o)
      inp_ids = tokenizer(inp, padding=True, truncation='longest_first', return_tensors='pt', max_length=512).input_ids.cuda()
      out_ids = tokenizer(out, return_tensors='pt').input_ids.cuda()
      loss = model(input_ids=inp_ids, labels=out_ids).loss
      loss.backward()
      #import ipdb
      #ipdb.set_trace()
      if (k + 1) % 16 == 0:
          optimizer.step()
          optimizer.zero_grad()
      #print((model.get_input_embeddings().weight[150,:]==initial_embedding[150,:]).min())
      #print((model.get_input_embeddings().weight[27569,:]==initial_embedding[27569,:]).max())
      total_loss = loss.item()
      count += 1
      pbar.update(1)
      pbar.set_postfix({'loss': total_loss/count})
      #wandb.log({'loss': loss.item()})

    #if epoch==0 or epoch==9:
    end = time.time()
    logging.info(end - start)
    logging.info(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
    model.save_pretrained(
    f'data/t5-base-100k_final_res_'+args.name)
    #epoch += 1











#Query-Doc-Relevan-true-false-Eos-colon

      #if (k>99450) and (k//99452==0):
      #    model.save_pretrained(f'data/t5-base--{args.query_token}-{args.doc_token}-{args.negs}-{args.seed}-{epoch}--{k}')
      

    
  #with _logger.duration(f'valid {epoch}'):
  #  reranker.model = model
  #  reranker.verbose = True
  #  res = reranker(valid_data)
  #  reranker.verbose = False
  #  metrics = {'epoch': epoch, 'loss': total_loss / count}
  #  metrics.update(pt.Utils.evaluate(res, valid_qrels, [nDCG, RR(rel=2)]))
  #  _logger.info(metrics)
  #  with open('log.jsonl', 'at') as f:
  #    f.write(json.dumps(metrics) + '\n')
   # if metrics['nDCG'] > max_ndcg:
   #   _logger.info('new best nDCG')
   #   model.save_pretrained(f'./data/t5-base-best-ndcg--{args.query_token}-{args.doc_token}-{args.negs}-{args.seed}-{epoch}')
   #   max_ndcg = metrics['nDCG']
   # wandb.log({"nDCG": metrics['nDCG']})
  
