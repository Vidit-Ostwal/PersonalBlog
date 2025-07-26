import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 4. Self-attention module
class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False) # shape: [d_model, d_model] (8, 8)
        self.Wk = nn.Linear(d_model, d_model, bias=False) # shape: [d_model, d_model] (8, 8)
        self.Wv = nn.Linear(d_model, d_model, bias=False) # shape: [d_model, d_model] (8, 8)

    def forward(self, x, mask = False):
        Q = self.Wq(x) # shape: [batch_size, seq_len, d_model] (1, 12, 8)
        K = self.Wk(x) # shape: [batch_size, seq_len, d_model] (1, 12, 8)
        V = self.Wv(x) # shape: [batch_size, seq_len, d_model] (1, 12, 8)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, seq_len, seq_len] (1, 12, 12)
        
        if mask:             
            seq_len = x.size(1)
            # Mask M: 0s in the lower triangle (visible), -inf in upper triangle (masked future tokens)
            M = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device)  # [seq_len, seq_len]
            M = M.masked_fill(M == 1, float('-inf')).masked_fill(M == 0, 0.0)
            scores = scores + M  # Add the mask *before* scaling

        scores = scores / d_k**0.5  # scale after adding the mask
        weights = F.softmax(scores, dim=-1)  # attention weights  # [batch_size, seq_len, seq_len] (1, 12, 12)
        output = torch.matmul(weights, V)  # shape: [batch_size, seq_len, d_model] (1, 12, 8)
        return scores, output, weights