import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape # (batch, time, channel)
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask future values
        weights = nn.functional.softmax(weights, dim=-1) 
        # weighted aggregation of values
        v = self.value(x)
        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(n_head)])
        self.linear = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.linear(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.network(x)

class Block(nn.Module):
    def __init__(self, n_head, n_embd, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.mha = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ff = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = self.mha(self.ln1(x)) + x # create residual connections
        x = self.ff(self.ln2(x)) + x 
        return x