###########################################
# Example: Generating Shakespeare-like text
###########################################

import torch
import torch.nn as nn
import pickle
from gpt import GPT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # utilize GPU for training
print(f'Using device: {device}')

##### DATA PROCESSING #####

# load and preprocess Shakespeare text 
path = 'example/shakespeare.txt'
with open(path, 'r') as f:
    text = f.read()
    text = text.lower()
    
print(f'WORD COUNT: {len(text.split())}')
print(f'SAMPLE:\n{text[:173]}\n...')

# encoding characters as integers
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_int = {c:i for i, c in enumerate(chars)}
encode = lambda s: [char_to_int[c] for c in s]
int_to_char = {i:c for i, c in enumerate(chars)}
decode = lambda l: ''.join([int_to_char[i] for i in l])

print(f'VOCAB SIZE: {vocab_size}')
print(f'ENCODING: {char_to_int}')

# split data into train and validation
data = torch.tensor(encode(text)).to(device)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

##### MODEL TRAINING #####

# hyperparameters 
batch_size = 128 # samples processed in parallel
block_size = 256 # max context length
train_iters = 2000
eval_interval = 100 
eval_iters = 200
learning_rate=1e-4
patience=5
best_val_loss=float('inf')
best_model_path='model_weights.pth'
steps_wo_improve=0

# train and evaluate model
model = GPT(vocab_size=vocab_size, 
            n_layer=8, # number of heads per block
            n_head=8, # number of attention blocks 
            n_embd=1024, # batchs size * number of heads 
            block_size=block_size, # max context length 
            dropout=0.3).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training helper functions
def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,), device=device) # generate random tensor of indices
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx]) # targets are one shift right from training
    return x, y

@torch.no_grad() # don't need to store gradients for evaluation
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training loop
for iter in range(train_iters+1): 
    # evaluation at intervals
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # check for improvement in val loss
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            steps_wo_improve = 0
            # save model
            torch.save(model.state_dict(), best_model_path)
        else:
            steps_wo_improve += 1
        # check convergence
        if steps_wo_improve >= patience:
            print('Training patience exceeded')
            break
    # model training
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model.load_state_dict(torch.load(best_model_path))
print(f'Training completed. Best model with val loss {best_val_loss:.4f}')
