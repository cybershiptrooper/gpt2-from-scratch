import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.cp_manager import load_model, load_latest_model
from utils.config import *

def process_tokens(tokens, tokenizer, config):
    # idk if this is a good way to do it but,
    # prompt too short : pad with eot
    # prompt too long : prompt <- prompt[-config.n_ctx:]
    # decode next word and replace pad every time/shift prompt
    prompt_tokens = tokens
    prompt_length = len(prompt_tokens)
    if(prompt_length < config.n_ctx):
        prompt_tokens =  prompt_tokens + [tokenizer.eot_token] * (config.n_ctx - len(prompt_tokens))
    if(prompt_length > config.n_ctx):
        prompt_tokens = prompt_tokens[-config.n_ctx:]
        prompt_length = config.n_ctx
    return prompt_tokens, prompt_length

def decode_prompt(model: torch.nn.Module, prompt, tokenizer, config, device="cpu", max_len=100, verbose=True):
    model.eval()
    result = []
    prompt_tokens_orig = tokenizer.encode(prompt)
    # pass to model and decode next word. 
    with torch.no_grad():
        for i in tqdm(range(max_len)):
            prompt_tokens, prompt_length = process_tokens(prompt_tokens_orig, tokenizer, config)
            # print(prompt_tokens, prompt_length)
            input = torch.tensor(prompt_tokens).unsqueeze(0).to(device)
            out = model(input)
            out.squeeze_(0)
            if(verbose):
                print("Out vector: ", out.argmax(dim=-1))
            # Predict the next word: 
            # out position when prompt_length < context = prompt_length
            # out position when prompt >= context = out[-1]
            out_pos = prompt_length-1 if prompt_length < config.n_ctx else -1
            max_of_out = out[out_pos].argmax(dim=-1)
            
            prompt_tokens_orig.append(max_of_out.item())  #append suggestion to prompt
            result.append(tokenizer.decode([max_of_out.item()]))
            if(verbose):
                print("Predicted word: {}, token: {}".format(tokenizer.decode([max_of_out.item()]), 
                                                        max_of_out.item()) )
    result = prompt + ' ' + ' '.join(result)

    return result
