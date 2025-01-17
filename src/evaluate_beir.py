import logging
import pathlib, os
import json
import torch
import sys
import transformers

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import DPR
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, AutoTokenizer, AutoModel
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from sentence_transformers import SentenceTransformer
from typing import Union, List, Dict, Tuple
from tqdm.autonotebook import trange
import torch

sys.path.append("./contriever")
sys.path.append("./contriever/src")
from contriever.src.contriever import Contriever
from contriever.src.beir_utils import DenseEncoderModel


class GTR:
    def __init__(self, model_path: Union[str, Tuple] = None, **kwargs):
        # Query tokenizer and model
        self.q_model = SentenceTransformer(model_path)
        self.q_tokenizer = self.q_model.tokenizer
        self.q_model.cuda()
        self.q_model.eval()

        # Context tokenizer and model
        self.ctx_tokenizer = self.q_tokenizer
        self.ctx_model = self.q_model

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> torch.Tensor:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):

                inputs = self.q_tokenizer(queries[start_idx:start_idx + batch_size], truncation=True, padding=True,
                                           return_tensors='pt').to("cuda")
                embeddings = self.q_model(inputs)["sentence_embedding"]

                query_embeddings += embeddings

        return torch.stack(query_embeddings)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> torch.Tensor:

        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(corpus), batch_size):
                texts = [f"{row['title']} {row['text']}" for row in corpus[start_idx:start_idx + batch_size]]
                inputs = self.ctx_tokenizer(texts, truncation='longest_first', padding=True,
                                             return_tensors='pt').to("cuda")
                embeddings = self.ctx_model(inputs)["sentence_embedding"]
                corpus_embeddings += embeddings

        return torch.stack(corpus_embeddings)


import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--model_code', type=str, default="contriever")
parser.add_argument('--score_function', type=str, default='dot',choices=['dot', 'cos_sim'])

parser.add_argument('--dataset', type=str, default="fiqa", help='BEIR dataset to evaluate')
parser.add_argument('--split', type=str, default='test')

parser.add_argument('--result_output', default="results/beir_results/tmp.json", type=str)

parser.add_argument("--per_gpu_batch_size", default=64, type=int, help="Batch size per GPU/CPU for indexing.")
parser.add_argument('--max_length', type=int, default=128)

args = parser.parse_args()

from utils import model_code_to_cmodel_name, model_code_to_qmodel_name

def compress(results):
    for y in results:
        k_old = len(results[y])
        break
    sub_results = {}
    for query_id in results:
        sims = list(results[query_id].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        sub_results[query_id] = {}
        for c_id, s in sims[:2000]:
            sub_results[query_id][c_id] = s
    for y in sub_results:
        k_new = len(sub_results[y])
        break
    logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
    return sub_results

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

logging.info(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### Download and load dataset
dataset = args.dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = os.path.join(out_dir, dataset)
if not os.path.exists(data_path):
    data_path = util.download_and_unzip(url, out_dir)
logging.info(data_path)

corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)
# corpus, queries, qrels = GenericDataLoader(data_path, corpus_file="corpus.jsonl", query_file="queries.jsonl").load(split=args.split)

logging.info("Loading model...")
if 'contriever' in args.model_code:
    encoder = Contriever.from_pretrained(model_code_to_cmodel_name[args.model_code]).cuda()
    tokenizer = transformers.BertTokenizerFast.from_pretrained(model_code_to_cmodel_name[args.model_code])
    model = DRES(DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer), batch_size=args.per_gpu_batch_size)
elif 'dpr' in args.model_code:
    model = DRES(DPR((model_code_to_qmodel_name[args.model_code], model_code_to_cmodel_name[args.model_code])), batch_size=args.per_gpu_batch_size, corpus_chunk_size=5000)
elif 'ance' in args.model_code:
    model = DRES(models.SentenceBERT(model_code_to_cmodel_name[args.model_code]), batch_size=args.per_gpu_batch_size)
elif 'gtr' in args.model_code:
    model = DRES(GTR(model_code_to_cmodel_name[args.model_code]), batch_size=args.per_gpu_batch_size)
else:
    raise NotImplementedError

logging.info(f"model: {model.model}")

retriever = EvaluateRetrieval(model, score_function=args.score_function, k_values=[1,3,5,10,20,100,1000]) # "cos_sim"  or "dot" for dot-product
results = retriever.retrieve(corpus, queries)

logging.info("Printing results to %s"%(args.result_output))
sub_results = compress(results)

with open(args.result_output, 'w') as f:
    json.dump(sub_results, f)
