#for dataset in nq msmarco hotpotqa fiqa trec-covid nfcorpus arguana quora scidocs fever scifact; do
#
#for model in contriever contriever-msmarco dpr-single dpr-multi ance; do

for dataset in nq; do

for model in gtr-base; do


FILE=results/beir_results/${dataset}-${model}.json

if ! [[ -f $FILE ]]; then
echo $FILE
mkdir -p results/beir_results
mkdir -p slurm_logs
sbatch --output=slurm_logs/%x-%j.out --error=slurm_logs/%x-%j.error --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=30G --partition=gpu_cuda --gres=gpu:a100:1 --account=a_ielab --job-name=${dataset}-${model} <<EOF
#!/bin/sh
source ~/.bashrc
conda activate corpus_poison

export TRANSFORMERS_CACHE=cache
python src/evaluate_beir.py --model_code $model --dataset $dataset --result_output $FILE
EOF
fi

done

done
