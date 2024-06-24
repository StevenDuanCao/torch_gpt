import torch
import torch.nn as nn
from attention import Block

class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.token_embd_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embd_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head, n_embd, block_size, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lin = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None): # idx: input tensor of tokens (Batch Size, Sequence Length)
        B, T = idx.shape
        tok_embd = self.token_embd_table(idx) # (B, T, C (vocab_size))
        pos_embd = self.pos_embd_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_embd + pos_embd # (B, T, C)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lin(x)
        
        if targets is None: # when generating text
            loss = None
        else: # when training model
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens): # idx: input tensor of tokens (Batch Size, Sequence Length)
        for _ in range(max_new_tokens): 
            idx_context = idx[:, -block_size:] # crop idx to context length
            logits, loss = self(idx_context) # get model predictions
            logits = logits[:,-1,:] # prediction of last time step
            probs = nn.functional.softmax(logits, dim=-1) # get probabilities of next token
            idx_next = torch.multinomial(probs, num_samples=1) # get next token
            idx = torch.cat((idx, idx_next), dim=1) # add next token to input
        return idx