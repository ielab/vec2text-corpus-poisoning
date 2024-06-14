
source activate vec2text

MODEL=gtr-base
DATASET=nq-train
OUTPUT_DIR=results/advp-v2t-gtr-k1000
k=1000
mkdir -p $OUTPUT_DIR
export OPENAI_API_KEY=temp
export HF_HOME=cache
export TRANSFORMERS_CACHE=cache
export VEC2TEXT_CACHE=cache/inversion

mkdir -p ${OUTPUT_DIR}
python src/attack_poison_v2t.py \
   --dataset ${DATASET} --split train \
   --model_code ${MODEL} \
   --output_dir ${OUTPUT_DIR} \
   --do_kmeans --k $k
