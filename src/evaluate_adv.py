import numpy as np

from beir import util
from beir.datasets.data_loader import GenericDataLoader

import os
import json
import sys

import argparse
import pytrec_eval

import torch
import copy


from tqdm import tqdm

from utils import load_models


def write_trec_run(run, file, name='adv', cut=1000):
    with open(file, 'w') as f:
        for qid in run:
            doc_score = run[qid]
            # sort by score
            doc_score = dict(sorted(doc_score.items(), key=lambda item: item[1], reverse=True)[:cut])
            for i, (doc, score) in enumerate(doc_score.items()):
                f.write(f'{qid} Q0 {doc} {i+1} {score} {name}\n')


def write_trec_qrel(qrels, file):
    with open(file, 'w') as f:
        for qid in qrels:
            for doc in qrels[qid]:
                f.write(f'{qid} 0 {doc} 1\n')


def evaluate_recall(results, qrels, k_values = [1,3,5,10,20,50,100,1000]):
    cnt = {k: 0 for k in k_values}
    for q in results:
        sims = list(results[q].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        gt = qrels[q]
        found = 0
        for i, (c, _) in enumerate(sims[:max(k_values)]):
            if c in gt:
                found = 1
            if (i + 1) in k_values:
                cnt[i + 1] += found
#             print(i, c, found)
    recall = {}
    for k in k_values:
        recall[f"Recall@{k}"] = round(cnt[k] / len(results), 5)
    
    return recall


def evaluate_ndcg10(results, qrels):
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {'ndcg_cut_10'})
    results = evaluator.evaluate(results)
    avg = 0
    for q in results:
        avg += results[q]['ndcg_cut_10']
    avg /= len(results)
    return {'NDCG@10': round(avg, 5)}


def main():
    parser = argparse.ArgumentParser(description='test')
    # The model and dataset used to generate adversarial passages 
    parser.add_argument("--attack_model_code", type=str, default="contriever", choices=["gtr-base", "gtr-base-v2t", "contriever-msmarco", "contriever", "dpr-single", "dpr-multi", "ance"])
    parser.add_argument("--attack_dataset", type=str, default="nq-train", choices=["nq-train", "msmarco", "nq"])
    parser.add_argument("--advp_path", type=str, default="results/advp", help="the path where generated adversarial passages are stored")
    parser.add_argument("--num_advp", type=str, default="50", help="how many adversarial passages are generated (i.e., k in k-means); you may test multiple by passing `--num_advp 1,10,50`")

    # The model and dataset used to evaluate the attack performance (e.g., if eval_model is different from attack_model, it studies attack across different models)
    parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["gtr-base", "gtr-base-v2t", "contriever-msmarco", "contriever", "dpr-single", "dpr-multi", "ance"])
    parser.add_argument('--eval_dataset', type=str, default="fiqa", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir eval_dataset')

    # Where to save the evaluation results (attack performance)
    parser.add_argument("--save_results", action='store_true', default=False)
    parser.add_argument("--eval_res_path", type=str, default="results/attack_results")

    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--pad_to_max_length', default=True)
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])

    args = parser.parse_args()

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.eval_dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.eval_dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        args.split = 'train'
    corpus, queries, qrels = data.load(split=args.split)

    if args.orig_beir_results is None:
        print("Please evaluate on BEIR first -- %s on %s"%(args.eval_model_code, args.eval_dataset))
        
        # Try to get beir eval results from ./beir_results
        print("Now try to get beir eval results from results/beir_results/...")
        if args.split == 'test':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
        elif args.split == 'dev':
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-dev.json"
        assert os.path.exists(args.orig_beir_results), f"Failed to get beir_results from {args.orig_beir_results}!"
        print(f"Automatically get beir_resutls from {args.orig_beir_results}.")
    
    with open(args.orig_beir_results, 'r') as f:
        results = json.load(f)
    
    assert len(qrels) == len(results)
    print('Total samples:', len(results))

    # Load models
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)

    model.eval()
    model.cuda()
    c_model.eval()
    c_model.cuda()

    def evaluate_adv(prefix, k, qrels, results):
        print('Prefix = %s, K = %d'%(prefix, k))
        adv_ps = []
        for s in range(k):
            file_name = "%s/%s-k%d-s%d.json"%(args.advp_path, prefix, k, s)
            if not os.path.exists(file_name):
                print(f"!!!!! {file_name} does not exist!")
                continue
            with open(file_name, 'r') as f:
                p = json.load(f)
                adv_ps.append(p)
        print('# adversaria passages', len(adv_ps))
        acc = 0
        tot = 0
        for s in range(len(adv_ps)):
            # print(s, adv_ps[s]["it"], adv_ps[s]["best_acc"], adv_ps[s]["tot"])
            acc += int(adv_ps[s]["tot"] * adv_ps[s]["best_acc"])
            tot += adv_ps[s]["tot"]
        # print("%.3f (%d / %d)"%(acc / tot, acc, tot))
        
        adv_results = copy.deepcopy(results)
        centroids_results = copy.deepcopy(results)
        
        adv_p_ids = [tokenizer.convert_tokens_to_ids(p["dummy"]) for p in adv_ps]

        try:
            adv_p_ids = torch.tensor(adv_p_ids).cuda()
            adv_attention = torch.ones_like(adv_p_ids, device='cuda')
            adv_token_type = torch.zeros_like(adv_p_ids, device='cuda')
            adv_input = {'input_ids': adv_p_ids, 'attention_mask': adv_attention, 'token_type_ids': adv_token_type}
        except ValueError:
            adv_passages = tokenizer.batch_decode(adv_p_ids)
            adv_input = tokenizer(adv_passages,
                                  padding=True,
                                  truncation=True,
                                  return_tensors="pt",
                                  add_special_tokens=False).to('cuda')
            adv_input['token_type_ids'] = torch.zeros_like(adv_input['input_ids'], device='cuda')

        with torch.no_grad():
            adv_embs = get_emb(c_model, adv_input)
        centroids = torch.load("%s/%s-k%d-centroids.pt"%(args.advp_path, prefix, k))
        if args.score_function == 'cos_sim':
            centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
        print("Adversarial passage distances to centroids:", torch.diagonal(torch.mm(adv_embs, centroids.T)))

        adv_qrels = {q: {"adv%d"%(s):1 for s in range(k)} for q in qrels}

        for i, query_id in tqdm(enumerate(results)):
            query_text = queries[query_id]
            query_input = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt")
            query_input = {key: value.cuda() for key, value in query_input.items()}
            with torch.no_grad():
                query_emb = get_emb(model, query_input)
                if args.score_function == 'cos_sim':
                    adv_sim = util.cos_sim(query_emb, adv_embs)
                    centroids_sim = util.cos_sim(query_emb, centroids)
                else:
                    adv_sim = torch.mm(query_emb, adv_embs.T)
                    centroids_sim = torch.mm(query_emb, centroids.T)
            
            for s in range(len(adv_ps)):
                adv_results[query_id]["adv%d"%(s)] = adv_sim[0][s].cpu().item()

            for s in range(len(adv_ps)):
                centroids_results[query_id]["adv%d"%(s)] = centroids_sim[0][s].cpu().item()

        # write_trec_run(adv_results, 'adv1000_run.txt')
        # write_trec_qrel(adv_qrels, 'adv1000_qrels.txt')

        adv_eval = evaluate_recall(adv_results, adv_qrels)
        oracle_eval = evaluate_recall(centroids_results, adv_qrels)
        print('Original NDCG@10', evaluate_ndcg10(results, qrels))
        print('Original Recall', evaluate_recall(results, qrels))
        print()
        print('Adversarial NDCG@10', evaluate_ndcg10(adv_results, adv_qrels))
        print('Adversarial Recall', adv_eval)
        print()
        print('Oracle NDCG@10', evaluate_ndcg10(centroids_results, adv_qrels))
        print('Oracle Recall', oracle_eval)
        return adv_eval
    
    mode = f"{args.attack_dataset}-{args.attack_model_code}"

    final_res = {}
    for k in args.num_advp.split(','):
        final_res[f"k={k}"] = evaluate_adv(mode, int(k), qrels, results)
    
    # print(f"Results: {final_res}")

    if args.save_results:
        # sub_dir: all eval results based on attack_model on attack_dataset with num_advp adversarial passages.
        sub_dir = '%s/%s-%s'%(args.eval_res_path, args.attack_dataset, args.attack_model_code)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)

        filename = '%s/%s-%s.json'%(sub_dir, args.eval_dataset, args.eval_model_code)
        if args.split == 'dev':
            filename = '%s/%s-%s-dev.json'%(sub_dir, args.eval_dataset, args.eval_model_code)

        print('Saving the results to %s'%(filename))
        with open(filename, 'w') as f:
            json.dump(final_res, f)


if __name__ == "__main__":
    main()