import os
import json
import argparse
from tqdm import tqdm
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
    parser.add_argument("--data-path", "-d", default="./data/datasets/ioi-t1/")
    parser.add_argument("--device", "-D", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", "-b", default=32, type=int)
    parser.add_argument("--num-examples", "-n", default=1000000, type=int)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load and convert ACDC circuit
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

    # Load model
    model = FPT2LMHeadModel.from_pretrained("gpt2", with_embedding_nodes=False).to(args.device)
    model.eval()
    model.load_all_log_alphas(edges)
    model.set_edge_threshold_for_deterministic(0.0)
    model.set_node_threshold_for_deterministic(0.0)

    control_model = FPT2LMHeadModel.from_pretrained("gpt2", with_embedding_nodes=False).to(args.device)
    control_model.reset_all_log_alphas()
    control_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data
    data = load_from_disk(args.data_path)["test"]
    if args.num_examples < len(data):
        data = data.select(range(args.num_examples))

    sentences = [data[i]['ioi_sentences'] for i in range(len(data))]
    corr_sentences = [data[i]['corr_ioi_sentences'] for i in range(len(data))]
    targets = []
    distractors = []
    prefix_lengths = []
    for i in range(len(data)):
        sentence = data[i]['ioi_sentences']
        prefix = sentence[:sentence.rfind(" ")]
        prefix_lengths.append(len(tokenizer.tokenize(prefix)) - 1)
        targets.append(tokenizer.encode(" " + data[i]['a'])[0])
        distractors.append(tokenizer.encode(" " + data[i]['b'])[0])

    # Eval
    accuracy = 0
    logit_difference = 0
    kl_divergence = 0
    exact_match = 0

    bar = tqdm(range(0, len(sentences), args.batch_size))
    for i in bar:
        n = min(args.batch_size, len(sentences) - i)
        input_ids = tokenizer(sentences[i:i+n], return_tensors="pt", padding=True).input_ids.to(args.device)
        corr_input_ids = tokenizer(corr_sentences[i:i+n], return_tensors="pt", padding=True).input_ids.to(args.device)

        with torch.no_grad():
            control_outputs = control_model(input_ids)
            corr_x = control_model(corr_input_ids, output_writer_states=True).writer_states.transpose(0, 1)
            outputs = model(input_ids, corr_x=corr_x)

        logits = outputs.logits
        control_logits = control_outputs.logits

        for j in range(n):
            pl = prefix_lengths[i+j]
            logit_t = logits[j, pl, targets[i+j]].item()
            logit_d = logits[j, pl, distractors[i+j]].item()
            logit_difference += logit_t - logit_d
            accuracy += (logits[j, pl].argmax() == targets[i+j]).int().item()
            exact_match += (logits[j, pl].argmax() == control_logits[j, pl].argmax()).int().item()
            log_p = torch.log_softmax(logits[j, pl], dim=-1)
            ref_log_p = torch.log_softmax(control_logits[j, pl], dim=-1)
            kl_divergence += torch.nn.functional.kl_div(log_p, ref_log_p, log_target=True, reduction="sum").item()

        bar.set_description(f"LD: {logit_difference/(i+n):.3f}, KL: {kl_divergence/(i+n):.3f}")

    n = len(sentences)
    overall_sparsity = model.get_effective_edge_sparsity()
    print(f"[i] Overall Edge Sparsity: {overall_sparsity}")
    print(f"[i]     Accuracy: {accuracy/n}")
    print(f"[i]     Logit difference: {logit_difference/n}")
    print(f"[i]     KL Divergence: {kl_divergence/n}")
    print(f"[i]     Exact Match: {exact_match/n}")

if __name__ == "__main__":
    main()
