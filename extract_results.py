import os
import re
import csv

joblog_dir = "/scratch/network/mc3803/Edge-Pruning/joblog"
results = []

def parse_n(filename):
    # Extract number of training examples from filename
    if "_n20" in filename: return 20
    elif "_n50" in filename: return 50
    elif "_n100" in filename: return 100
    else: return 200  # baseline

def parse_task(filename):
    if "eval_ioi_t1" in filename: return "ioi_t1"
    elif "eval_ioi" in filename: return "ioi"
    elif "eval_gt" in filename: return "gt"
    elif "eval_gp" in filename: return "gp"
    return None

def parse_es(filename):
    match = re.search(r'es(\d+\.\d+)', filename)
    return float(match.group(1)) if match else None

for folder in [joblog_dir, os.path.join(joblog_dir, "baseline_evals")]:
    for fname in os.listdir(folder):
        if not fname.startswith("eval_") or not fname.endswith(".out"):
            continue
        task = parse_task(fname)
        if task is None:
            continue
        n = parse_n(fname)
        es = parse_es(fname)
        
        fpath = os.path.join(folder, fname)
        with open(fpath) as f:
            content = f.read()
        
        kl = re.search(r'KL Divergence: ([0-9.]+)', content)
        ld = re.search(r'Logit difference: ([0-9.\-]+)', content)
        pd = re.search(r'Probability difference: ([0-9.]+)', content)
        actual_es = re.search(r'Overall Edge Sparsity: ([0-9.]+)', content)
        
        results.append({
            "task": task,
            "n_train": n,
            "target_es": es,
            "actual_es": float(actual_es.group(1)) if actual_es else None,
            "kl": float(kl.group(1)) if kl else None,
            "logit_diff": float(ld.group(1)) if ld else None,
            "prob_diff": float(pd.group(1)) if pd else None,
        })

# Sort by task, n_train, actual_es
results.sort(key=lambda x: (x["task"], x["n_train"], x["actual_es"] or 0))

# Write CSV
with open("/scratch/network/mc3803/Edge-Pruning/results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["task", "n_train", "target_es", "actual_es", "kl", "logit_diff", "prob_diff"])
    writer.writeheader()
    writer.writerows(results)

print(f"Extracted {len(results)} results to results.csv")
