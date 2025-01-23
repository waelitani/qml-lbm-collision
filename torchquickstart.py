import os
import torch
torch.set_default_device('cuda')

def train(dataloader, model, loss_fn, optimizer, c, rank, world_size):
    
    size = len(dataloader.dataset)
    batch = 0
    for X, y in dataloader:
        
        optimizer.zero_grad()
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
       
        # Backpropagation        
        loss.backward(retain_graph = True)
        
        optimizer.step()
        
        loss, current = loss.item(), (batch + 1) * world_size*len(X)
        batch += 1
        if rank == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}({len(X):>5d})/{size:>5d}]\n")
            with open("log.out", "a") as thisfile:
                thisfile.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
    return loss
    
    