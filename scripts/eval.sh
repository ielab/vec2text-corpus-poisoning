source activate vec2text

export HF_HOME=cache
export TRANSFORMERS_CACHE=cache

EVAL_MODEL=gtr-base
EVAL_DATASET=nq
ATTK_MODEL=gtr-base
ATTK_DATASET=nq-train
mkdir -p results/beir_results

FILE=results/beir_results/${EVAL_DATASET}-${EVAL_MODEL}.json

python src/evaluate_beir.py --per_gpu_batch_size 512 --model_code ${EVAL_MODEL} --dataset ${EVAL_DATASET} --result_output ${FILE}

python src/evaluate_adv.py --save_results \
   --attack_model_code ${ATTK_MODEL} --attack_dataset ${ATTK_DATASET} \
   --advp_path results/advp --num_advp 10 \
   --eval_model_code ${EVAL_MODEL} --eval_dataset ${EVAL_DATASET} \
   --orig_beir_results results/beir_results/${EVAL_DATASET}-${EVAL_MODEL}.json

ATTK_MODEL=gtr-base
k=10
advp_path=results/advp-v2t-gtr-k10
eval_res_path=results/attack_results_v2t-gtr-k10
python src/evaluate_adv.py --save_results \
   --attack_model_code ${ATTK_MODEL} --attack_dataset ${ATTK_DATASET} \
   --advp_path ${advp_path} --num_advp ${k} \
   --eval_model_code ${EVAL_MODEL} --eval_dataset ${EVAL_DATASET} \
   --orig_beir_results results/beir_results/${EVAL_DATASET}-${EVAL_MODEL}.json \
   --eval_res_path ${eval_res_path}


ATTK_MODEL=gtr-base
k=10
advp_path=results/advp
eval_res_path=results/attack_results
python src/evaluate_adv.py --save_results \
   --attack_model_code ${ATTK_MODEL} --attack_dataset ${ATTK_DATASET} \
   --advp_path ${advp_path} --num_advp ${k} \
   --eval_model_code ${EVAL_MODEL} --eval_dataset ${EVAL_DATASET} \
   --orig_beir_results results/beir_results/${EVAL_DATASET}-${EVAL_MODEL}.json \
   --eval_res_path ${eval_res_path}