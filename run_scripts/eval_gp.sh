for dir in /scratch/network/mc3803/Edge-Pruning/data/runs/gp-wo_node_loss*/; do
    sbatch --partition=gpu --gres=gpu:1 --mem=30G --time=00:30:00 --constraint=gpu80 \
        --job-name=eval_gp \
        --output=/scratch/network/mc3803/Edge-Pruning/joblog/eval_gp_$(basename $dir).out \
        --wrap="cd /scratch/network/mc3803/Edge-Pruning && \
        module load anaconda3/2025.12 && \
        conda activate /scratch/network/mc3803/envs/moa && \
        export TRANSFORMERS_OFFLINE=1 && \
        export HF_HOME=/scratch/network/mc3803/.cache/huggingface && \
        python src/eval/gp.py -m $dir -w"
done
