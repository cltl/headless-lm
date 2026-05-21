import argparse
import torch
from transformers import BertForMaskedLM, BertConfig, GPTNeoXConfig, GPTNeoXForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--ckpt_path")
parser.add_argument("-o", "--output_dir")
parser.add_argument("-m", "--mode", default="mlm")
args = parser.parse_args()

ckpt_path = args.ckpt_path
output_dir = args.output_dir
mode = args.mode
# ----------------------------
# 1. Load clean checkpoint
# ----------------------------
checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint["state_dict"]


# ----------------------------
# 2. Strip Lightning prefixes
# ----------------------------
hf_state_dict = {}

for k, v in state_dict.items():
    if k.startswith(f"task.{mode}_model."):
        new_key = k[len(f"task.{mode}_model.") :]
        hf_state_dict[new_key] = v


print(f"Converted {len(hf_state_dict)} parameters")


# ----------------------------
# 3. Load config
# ----------------------------
# IMPORTANT: match your training config!
# If you changed architecture, override here:
# config.hidden_size = ...
if mode == "mlm":
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.max_position_embeddings = 128
    config.vocab_size = 50000
    # 4. Initialize model
    model = BertForMaskedLM(config)
else:
    config = GPTNeoXConfig.from_pretrained("EleutherAI/pythia-70m")
    model = GPTNeoXForCausalLM(config)


# ----------------------------
# 5. Load weights
# ----------------------------
missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)

print("\n=== LOAD REPORT ===")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)


# ----------------------------
# 6. Save for HuggingFace
# ----------------------------
# output_dir = "./hf_model"

model.save_pretrained(output_dir)

print(f"\n✅ Model saved to {output_dir}")
