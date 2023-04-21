import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.cp_manager import load_model, load_latest_model
import tiktoken as TT
from utils.config import *

def decode_prompt(model: torch.nn.Module, prompt, tokenizer, config, device="cpu", max_len=100):
    model.eval()
    result = []
    
    prompt_tokens = tokenizer.encode(prompt)
    prompt_length = len(prompt_tokens)
    print(prompt_tokens)
    # idk if this is a good way to do it but,
    # prompt too short : pad with eot
    # prompt too long : prompt <- prompt[-config.n_ctx:]
    # decode next word and replace pad every time/shift prompt
    if(prompt_length < config.n_ctx):
        prompt_tokens =  prompt_tokens + [tokenizer.eot_token] * (config.n_ctx - len(prompt_tokens))
    # pass to model and decode next word. 
    with torch.no_grad():
        for i in tqdm(range(max_len)):
            if(prompt_length > config.n_ctx):
                prompt_tokens = prompt_tokens[-config.n_ctx:]
                prompt_length = config.n_ctx
            input = torch.tensor(prompt_tokens).unsqueeze(0).to(device)
            out = model(input)
            out.squeeze_(0)
            print(out.argmax(dim=-1))
            # Predict the next word: 
            # for a single word prompt, length = 1
            # first out position = 1 + 0 - 1 (length + i - 1)
            out = out[prompt_length+i-1].argmax(dim=-1)
            prompt_tokens.append(out.item())
            result.append(tokenizer.decode([out.item()]))
            print(tokenizer.decode([out.item()]), out.item())
    result = prompt + ' ' + ' '.join(result)

    return result
