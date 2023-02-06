import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from pathlib import Path
import pandas as pd

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, BatchSampler

__all__ = ['accuracy', 'report', 'Dataset', 'fit', 'get_dls']

def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()

def report(loss, preds, yb): print(f"{loss:.4f}, {accuracy(preds, yb):.4f}")

class Dataset:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def fit(model, opt, loss_fn, train_dl, valid_dl, epochs=3):
    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl:
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            tot_loss, tot_acc, count = 0., 0., 0.
            for xb,yb in valid_dl:
                out = model(xb)
                loss = loss_fn(out, yb)
                acc = accuracy(out, yb)
                n = len(xb)
                count += n
                tot_loss += loss.item()*n # we *n because it returns an average by default (think of it as spreading it out the batch)
                tot_acc += acc.item()*n
        print(epoch, tot_loss/count, tot_acc/count)
    return tot_loss/count, tot_acc/count

def get_dls(train_ds, valid_ds, bs, **kwargs):
    return(
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs*2, shuffle=False, **kwargs))