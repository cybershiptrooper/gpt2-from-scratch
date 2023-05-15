import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.config import defaultConfig
from utils.cp_manager import save_model

def plot(losses):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()

def eval_net(model, dataset, batch_size=32, config=defaultConfig):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    context = dataset.block_size
    with torch.no_grad():
        loss = 0
        accuracy = 0
        for batch in tqdm(dataloader, total=len(dataloader)):
            x, y = batch
            y = torch.concat([x[:, 1:], y.unsqueeze(1)], dim=1)
            out = model(x)
            if(config.logits):
                out = F.log_softmax(out, dim=-1) 
            else:
                out = torch.log(out)
            loss -= out.gather(
                    dim=-1, index=y[..., None]
                ).squeeze().mean() 
            preds = out.argmax(dim=-1)
            # print(preds.shape, y.shape, (preds == y).sum().item(), sep="\n===============\n===============\n===============\n===============\n")
            accuracy += (preds == y).sum().item()
    model.train()
    return {"loss": loss.item() / len(dataset), "accuracy": accuracy/(len(dataset) * context ) }

def train_net(model, train_dataset, val_dataset, optimizer, epochs=2, 
              batch_size=32, config=defaultConfig, 
              print_every=10, save = True, save_every = 50, plot_losses = True, save_message=""):
    losses = []
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for i, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
            x, y = batch
            y = torch.concat([x[:, 1:], y.unsqueeze(1)], dim=1)

            optimizer.zero_grad()
            out = model(x)
            # nll loss
            if(config.logits):
                out = F.log_softmax(out, dim=-1) 
            else:
                out = torch.log(out)
            loss = -out.gather(
                    dim=-1, index=y[..., None]
                ).squeeze().mean() # should I add a hyperparam for y vs shifted block instead?
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if((i) % print_every == 0):
                print(f"Epoch {epoch+1}/{epochs}, \
                      Step {i+1}/{len(train_dataloader)} \
                      loss: {loss.item():.4f}")
                print(out.argmax(dim=-1))
            if((i) % save_every == 0 and save ):
                val_results = eval_net(model, val_dataset, batch_size, config)
                print("val results: ", val_results)
                save_model(model, optimizer, config, 
                           file_name=f"{save_message}.epoch_{epoch+1}_step__{i}__acc_{val_results['accuracy']}.pt" )
        print(f"Epoch {epoch+1}/{epochs} loss: {loss.item():.4f}")
        print("val results: ", eval_net(model, val_dataset, batch_size, config))

    if(plot_losses):
        plot(losses)
