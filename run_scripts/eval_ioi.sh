for dir in /scratch/network/mc3803/Edge-Pruning/data/runs/ioi-wo_node_loss*/; do
    sbatch --partition=gpu --gres=gpu:1 --mem=30G --time=00:30:00 \
        --job-name=eval_ioi \
        --output=/scratch/network/mc3803/Edge-Pruning/joblog/eval_ioi_$(basename $dir).out \
        --wrap="cd /scratch/network/mc3803/Edge-Pruning && \
        module load anaconda3/2025.12 && \
        conda activate /scratch/network/mc3803/envs/moa && \
        export TRANSFORMERS_OFFLINE=1 && \
        export HF_HOME=/scratch/network/mc3803/.cache/huggingface && \
        python src/eval/ioi.py -m $dir -w"
done
