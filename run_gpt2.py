from utils.inference import decode_prompt
from utils.cp_manager import load_latest_model, load_latest_model_config
import tiktoken as TT
from model.transformer_model import TransformerModel
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Muzaffar was a hunter.")
    parser.add_argument("--ans_tokens", type=int, default=100)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
    latest_config = load_latest_model_config()
    model = TransformerModel(config=latest_config)
    load_latest_model(model)
    model.to(device)
    tokenizer = TT.get_encoding("gpt2")
    print(decode_prompt(model, args.prompt, tokenizer, latest_config, device=device, max_len=args.ans_tokens))