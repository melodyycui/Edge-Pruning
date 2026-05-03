# src/eval/acdc_to_checkpoint.py
import os
import json
import argparse
import torch
import sys
sys.path.append(os.path.join(os.getcwd(), "src/modeling/"))
from modeling_fpt2 import FPT2LMHeadModel, writer_name_to_idx

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
    parser.add_argument("--out-dir", "-o", default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.out_dir is None:
        basename = os.path.basename(args.acdc_json_path)
        print(f"basename: {basename}")
        task = basename.replace('-graph.json', '')
        print(f"task: {task}")
        args.out_dir = f"data/acdc_checkpoints/{task}/"
        print(f"out_dir: {args.out_dir}")

    # Load ACDC circuit
    with open(args.acdc_json_path) as f:
        data = json.load(f)

    edges = []
    for edge in data["original"]:
        try:
            writer = convert_node_name_from(edge["from"])
            reader = convert_node_name_to(edge["to"])
            edges.append((writer, reader))
        except Exception as e:
            print(f"Skipping {edge}: {e}")

    print(f"Loaded {len(edges)} edges")

    # Load model and set all log_alphas to -10 (prune everything)
    model = FPT2LMHeadModel.from_pretrained("gpt2", with_embedding_nodes=False)
    
    # Set all to -10 first
    with torch.no_grad():
        for layer in model.transformer.h:
            layer.q_read_log_alphas.fill_(-10.0)
            layer.k_read_log_alphas.fill_(-10.0)
            layer.v_read_log_alphas.fill_(-10.0)
            layer.attn_write_log_alphas.fill_(10.0)  # keep all nodes
            layer.mlp_read_log_alphas.fill_(-10.0)
            layer.mlp_write_log_alphas.fill_(10.0)   # keep all nodes
        model.transformer.final_read_log_alphas.fill_(-10.0)

    # Now set +10 for edges in circuit
    # We need writer_name_to_idx and similar from the model
    # Use the existing infrastructure
    num_heads = model.transformer.h[0].attn.num_heads
    num_layers = len(model.transformer.h)

    with torch.no_grad():
        for writer, reader in edges:
            try:
                if reader == "resid_post":
                    # final_read_log_alphas
                    w_idx = writer_name_to_idx(writer, num_layers=num_layers, num_heads=num_heads, with_embedding_nodes=False)
                    model.transformer.final_read_log_alphas.data[w_idx] = 10.0
                elif reader.startswith("m"):
                    layer_idx = int(reader[1:])
                    w_idx = writer_name_to_idx(writer, num_layers=num_layers, num_heads=num_heads, with_embedding_nodes=False)
                    model.transformer.h[layer_idx].mlp_read_log_alphas.data[w_idx] = 10.0
                elif reader.startswith("a"):
                    parts = reader.split(".")
                    layer_idx = int(parts[0][1:])
                    head_idx = int(parts[1][1:])
                    qkv = parts[2]
                    w_idx = writer_name_to_idx(writer, num_layers=num_layers, num_heads=num_heads, with_embedding_nodes=False)
                    if qkv == "q":
                        model.transformer.h[layer_idx].q_read_log_alphas.data[w_idx, head_idx] = 10.0
                    elif qkv == "k":
                        model.transformer.h[layer_idx].k_read_log_alphas.data[w_idx, head_idx] = 10.0
                    elif qkv == "v":
                        model.transformer.h[layer_idx].v_read_log_alphas.data[w_idx, head_idx] = 10.0
            except Exception as e:
                print(f"Error processing edge {writer}->{reader}: {e}")
    
    print(f"About to makedirs with: {args.out_dir!r}")
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    print(f"Saved checkpoint to {args.out_dir}")

if __name__ == "__main__":
    main()