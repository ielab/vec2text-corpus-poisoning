MODEL=$1
DATASET=$2
OUTPUT_PATH=results/advp
k=$3

mkdir -p $OUTPUT_PATH

for s in $(eval echo "{0..$((k-1))}"); do

sbatch --output=slurm_logs/%x-%j.poison.out --error=slurm_logs/%x-%j.poison.error --nodes=1 --time=20:00:00 --ntasks-per-node=1 --cpus-per-task=4 --mem=30G --partition=gpu_cuda --gres=gpu:h100:1 --account=a_ielab --job-name=poison-${s} <<EOF
#!/bin/sh
source ~/.bashrc
conda activate vec2text

export TRANSFORMERS_CACHE=cache
python src/attack_poison.py \
   --dataset ${DATASET} --split train \
   --model_code ${MODEL} \
   --num_cand 100 --per_gpu_eval_batch_size 64 --num_iter 5000 --num_grad_iter 1 \
   --output_file ${OUTPUT_PATH}/${DATASET}-${MODEL}-k${k}-s${s}.json \
   --do_kmeans --k $k --kmeans_split $s --dont_init_gold
EOF

done
