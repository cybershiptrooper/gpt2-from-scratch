import torch
import os
from model.transformer_model import TransformerModel

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../checkpoints")

def save_model(model, optimizer, config, file_name):
    if(not file_name.endswith(".pt")):
        file_name += ".pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, os.path.join(dir_path, file_name))

    # store latest model name to latest.log
    with open(os.path.join(dir_path, "latest.log"), 'w') as f:
        f.write(file_name)

def load_model(model=None, file_name="", optimizer=None):
    if(not file_name.endswith(".pt")):
        file_name += ".pt"
    checkpoint = torch.load(os.path.join(dir_path, file_name))
    
    if not os.path.exists(os.path.join(dir_path, file_name)):
        print("Model does not exist. Loading latest model instead.")
        load_latest_model()
    if model is None:
        model = TransformerModel(config=checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    if(optimizer is not None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def load_config(file_name):
    if not os.path.exists(os.path.join(dir_path, file_name)):
        print("Model does not exist. Loading latest model instead.")
        load_latest_model_config()
    checkpoint = torch.load(os.path.join(dir_path, file_name))
    return checkpoint['config']

def load_latest_model(model=None, optimizer=None):
    if not os.path.exists(os.path.join(dir_path, "latest.log")):
        raise ValueError("latest.log does not exist")
    with open(os.path.join(dir_path, "latest.log"), 'r') as f:
        file_name = f.read()
        if(not file_name.endswith(".pt")):
            file_name += ".pt"
    return load_model(model, file_name, optimizer=optimizer)

def load_latest_model_config():
    if not os.path.exists(os.path.join(dir_path, "latest.log")):
        raise ValueError("latest.log does not exist")
    with open(os.path.join(dir_path, "latest.log"), 'r') as f:
        file_name = f.read()
        if(not file_name.endswith(".pt")):
            file_name += ".pt"
    return load_config(file_name)