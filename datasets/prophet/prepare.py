import os
import requests
import tiktoken as TT
import numpy as np

# used https://github.com/karpathy/nanoGPT with minor changes
dir_path = os.path.dirname(os.path.realpath(__file__))

def prepare(tokenizer_name):
    # reason? I love the way Kahlil Gibran wrote his stuff :)
    input_file_path = os.path.join(dir_path, 'input.txt')
    train_path = os.path.join(dir_path, f'bin/train-{tokenizer_name}.bin')
    val_path = os.path.join(dir_path, f'bin/val-{tokenizer_name}.bin')
    for file in [train_path, val_path]:
        if(os.path.exists(file)):
            return
    # if not os.path.exists(input_file_path):
    #     data_url = 'https://www.gutenberg.org/cache/epub/58585/pg58585.txt'
    #     with open(input_file_path, 'w') as f:
    #         f.write(requests.get(data_url).text)

    with open(input_file_path, 'r') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode data
    enc = TT.get_encoding(tokenizer_name)
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens") #16,417
    print(f"val has {len(val_ids):,} tokens") #1,811

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
