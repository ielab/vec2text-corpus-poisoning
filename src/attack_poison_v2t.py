import logging
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import json
import random
from tqdm import tqdm
from tqdm.autonotebook import trange

from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
import torch.nn.functional as F
import vec2text

from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, T5PreTrainedModel, DataCollatorForSeq2Seq
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

import argparse
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from collections import Counter

from utils import load_models
import time


def evaluate_acc(model, c_model, get_emb, dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type,
                 data_collator, device='cuda'):
    """Returns the 2-way classification accuracy (used during training)"""
    model.eval()
    c_model.eval()
    acc = 0
    tot = 0
    for idx, (data) in tqdm(enumerate(dataloader), desc='Evaluating'):
        data = data_collator(data)  # [bsz, 3, max_len]

        # Get query embeddings
        q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
        q_emb = get_emb(model, q_sent)  # [b x d]

        gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
        gold_emb = get_emb(c_model, gold_pass)  # [b x d]

        sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()

        p_sent = {'input_ids': adv_passage_ids,
                  'attention_mask': adv_passage_attention,
                  'token_type_ids': adv_passage_token_type}
        p_emb = get_emb(c_model, p_sent)  # [k x d]

        sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]

        acc += (sim_to_gold > sim).sum().cpu().item()
        tot += q_emb.shape[0]

    print(f'Acc = {acc / tot * 100} ({acc} / {tot})')
    return acc / tot


def kmeans_split(data_dict, model, get_emb, tokenizer, k):
    """Get all query embeddings and perform kmeans"""
    # get query embs
    q_embs = []
    with torch.no_grad():
        for start_idx in trange(0, len(data_dict["sent0"]), 128, desc='Encoding all train queries'):
            query_input = tokenizer(data_dict["sent0"][start_idx:start_idx + 128],
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt").to('cuda')
            query_embs = get_emb(model, query_input)
            q_embs.extend([emb.cpu().numpy() for emb in query_embs])
    q_embs = np.array(q_embs)
    print("q_embs", q_embs.shape)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(q_embs)
    return kmeans.cluster_centers_


def main():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--model_code', type=str, default='contriever')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--pad_to_max_length', default=True)

    parser.add_argument("--output_dir", default=None, type=str)

    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--kmeans_split", default=0, type=int)
    parser.add_argument("--do_kmeans", default=False, action="store_true")

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(0)

    # Load models
    model, c_model, tokenizer, get_emb = load_models(args.model_code)

    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device)
    inversion_model = vec2text.models.InversionModel.from_pretrained(
        'xxx/gtr_base_st-embed/checkpoint-133375').to(device)
    correct_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
        'xxx/gtr_base_st-embed-corrector-1/checkpoint-348000').to(
        device)

    inversion_trainer = vec2text.trainers.InversionTrainer(
        model=inversion_model,
        train_dataset=None,
        eval_dataset=None,
        data_collator=DataCollatorForSeq2Seq(
            inversion_model.tokenizer,
            label_pad_token_id=-100,
        ),
    )

    # backwards compatibility stuff
    correct_model.config.dispatch_batches = None
    corrector = vec2text.trainers.Corrector(
        model=correct_model,
        inversion_trainer=inversion_trainer,
        args=None,
        data_collator=vec2text.collator.DataCollatorForCorrection(
            tokenizer=inversion_trainer.model.tokenizer
        ),
    )
    # Load datasets
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        args.split = 'train'
    corpus, queries, qrels = data.load(split=args.split)
    l = list(qrels.items())
    random.shuffle(l)
    qrels = dict(l)

    data_dict = {"sent0": [], "sent1": []}
    for q in qrels:
        q_ctx = queries[q]
        for c in qrels[q]:
            c_ctx = corpus[c].get("title") + ' ' + corpus[c].get("text")
            data_dict["sent0"].append(q_ctx)
            data_dict["sent1"].append(c_ctx)

    centroids = kmeans_split(data_dict, model, get_emb, tokenizer, k=args.k)
    centroids = torch.tensor(centroids).to(device)
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
    torch.save(centroids, os.path.join(args.output_dir, f'{args.dataset}-{args.model_code}-k{args.k}-centroids.pt'))

    adv_passages = []
    # for chunk of centorids
    for i in tqdm(range(0, len(centroids), 10)):
        chunk_centroids = centroids[i:i+10]

        results = vec2text.invert_embeddings(
            embeddings=chunk_centroids,
            corrector=corrector,
            num_steps=50,
            sequence_beam_width=3
        )
        adv_passages.extend(results)

    # inverter only
    # inverter = vec2text.models.InversionModel.from_pretrained(
    #     '/scratch/project/neural_ir/arvin/vec2text-reproduce/saves/checkpoint-93750').to('cuda')
    # generation = inverter.generate({'frozen_embeddings': centroids}, {})
    # adv_passages = tokenizer.batch_decode(generation, skip_special_tokens=True)

    with torch.no_grad():
        adv_passage_input = tokenizer(adv_passages,
                                      padding=True,
                                      truncation=True,
                                      return_tensors="pt").to('cuda')
        adv_passage_emb = get_emb(c_model, adv_passage_input)

    cos_sim = []
    for adv_emb, centroid in zip(adv_passage_emb, centroids):
        cos_sim.append(torch.nn.CosineSimilarity(dim=0, eps=1e-6)(adv_emb, centroid).item())

    if args.output_dir is not None:
        for i, adv_pass in enumerate(adv_passages):
            with open(os.path.join(args.output_dir, f'{args.dataset}-{args.model_code}-k{args.k}-s{i}.json'), 'w') as f:
                json.dump(
                    {"it": 0, "best_acc": cos_sim[i],
                     "dummy": tokenizer.convert_ids_to_tokens(tokenizer(adv_pass)['input_ids']),
                     "tot": 1}, f)


if __name__ == "__main__":
    main()
