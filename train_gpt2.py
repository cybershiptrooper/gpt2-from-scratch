import torch
import torch.nn as nn
from model.transformer_model import TransformerModel
from datasets.prophet.dataset import ProphetDataset
from utils.config import *
from utils.train import train_net

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu'
    torch.manual_seed(1) # for reproducibility
    config = GPT2Config(
        debug=False,
        logits=True,
    )
    print(config)
    model = TransformerModel(config).to(device)
    if device == "cuda":
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
        print("done compiling")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_dataset = ProphetDataset('train', 'gpt2', device, config.n_ctx)
    val_dataset = ProphetDataset('val', 'gpt2', device, config.n_ctx)
    train_net(model, train_dataset, val_dataset, 
              optimizer, epochs=1, batch_size=2, config=config, print_every=1)