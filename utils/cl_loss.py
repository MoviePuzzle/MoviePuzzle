import torch
import torch.nn as nn

def cl_loss(anchor, in_item, out_item):
    # input [B, hidden_size] -> [B]
    in_hs = torch.einsum('ij,ij->i', [anchor, in_item])
    out_hs = torch.einsum('ij,ij->i', [anchor, out_item])
    loss_func = nn.LogSoftmax(dim=0)
    hs = torch.stack([in_hs, out_hs], dim=0)
    ls = loss_func(hs)[0]
    return -ls