import os
import json
import argparse
from tqdm import tqdm
from scipy.stats import kendalltau
import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
import sys
sys.path.append(os.path.join(os.getcwd(), "src/modeling/"))
from modeling_fpt2 import FPT2LMHeadModel

def convert_node_name_from(name):
    parts = name.split(".")
    if parts[0] == "head":
        return f"a{parts[1]}.h{parts[2]}"
    elif parts[0] == "mlp":
        return f"m{parts[1]}"
    else:
        raise ValueError(f"Unknown from name: {name}")

def convert_node_name_to(name):
    parts = name.split(".")
    if parts[0] == "head":
        return f"a{parts[1]}.h{parts[2]}.{parts[3]}"
    elif parts[0] == "mlp":
        return f"m{parts[1]}"
    elif name == "resid_post":
        return "resid_post"
    else:
        raise ValueError(f"Unknown to name: {name}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc-json-path", "-a", required=True)
    parser.add_argument("--model-path", "-m", required=True)
    parser.add_argument("--data-path", "-d", default="./data/datasets/gt/")
    parser.add_argument("--device", "-D", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", "-b", default=32, type=int)
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.acdc_json_path) as f:
        data = json.load(f)
    edges = []
    for edge in data["original"]:
        try:
            writer = convert_node_name_from(edge["from"])
            reader = convert_node_name_to(edge["to"])
            edges.append((writer, reader))
        except ValueError as e:
            print(f"Skipping edge {edge}: {e}")

    print(f"[i] Loaded {len(edges)} edges from ACDC circuit")

    model = FPT2LMHeadModel.from_pretrained(args.model_path, with_embedding_nodes=False).to(args.device)
    model.eval()
    model.set_edge_threshold_for_deterministic(0.0)
    model.set_node_threshold_for_deterministic(0.0)

    control_model = FPT2LMHeadModel.from_pretrained("gpt2", with_embedding_nodes=False).to(args.device)
    control_model.reset_all_log_alphas()
    control_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    all_digits = torch.LongTensor([tokenizer.encode("{:02d}".format(i))[0] for i in range(100)]).to(args.device)

    dataset = load_from_disk(args.data_path)["test"]
    sentences = [dataset[i]['prefix'] for i in range(len(dataset))]
    corr_sentences = [dataset[i]['corr_prefix'] for i in range(len(dataset))]
    digits = [int(dataset[i]['digits']) for i in range(len(dataset))]

    prob_diff = 0
    prob_diff_10 = 0
    kt = 0
    kl = 0

    bar = tqdm(range(0, len(sentences), args.batch_size))
    for i in bar:
        n = min(args.batch_size, len(sentences) - i)
        input_ids = tokenizer(sentences[i:i+n], return_tensors="pt", padding=True).input_ids.to(args.device)
        corr_input_ids = tokenizer(corr_sentences[i:i+n], return_tensors="pt", padding=True).input_ids.to(args.device)

        with torch.no_grad():
            control_outputs = control_model(input_ids)
            corr_x = control_model(corr_input_ids, output_writer_states=True).writer_states.transpose(0, 1)
            outputs = model(input_ids, corr_x=corr_x)

        logits = outputs.logits[:, -1, :]
        control_logits = control_outputs.logits[:, -1, :]

        for j in range(n):
            circuit_digit_logits = logits[j, all_digits]
            ref_digit_logits = control_logits[j, all_digits]
            probs = torch.softmax(circuit_digit_logits, dim=-1)
            ref_probs = torch.softmax(ref_digit_logits, dim=-1)
            d = digits[i+j]
            prob_diff += (probs[d+1:].sum() - probs[:d].sum()).item()
            l = max(0, d-10)
            r = min(100, d+11)
            prob_diff_10 += (probs[d+1:r].sum() - probs[l:d].sum()).item()
            kt += kendalltau(probs.cpu().numpy(), ref_probs.cpu().numpy()).correlation
            log_p = torch.log_softmax(logits[j], dim=-1)
            ref_log_p = torch.log_softmax(control_logits[j], dim=-1)
            kl += torch.nn.functional.kl_div(log_p, ref_log_p, log_target=True, reduction="sum").item()

    n = len(sentences)
    overall_sparsity = model.get_effective_edge_sparsity()
    print(f"[i] Overall Edge Sparsity: {overall_sparsity}")
    print(f"[i]     Probability difference: {prob_diff/n}")
    print(f"[i]     Probability difference (10): {prob_diff_10/n}")
    print(f"[i]     Kendall's Tau: {kt/n}")
    print(f"[i]     KL Divergence: {kl/n}")

if __name__ == "__main__":
    main()