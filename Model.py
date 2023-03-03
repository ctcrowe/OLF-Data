import torch
import torch.nn as nn
import math
import os
from torch.nn import functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass

# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 16 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2
# ------------


chars = [' ', ',', '_', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

outputs = ['0', '5', '7', '11', '15', '20', '30','35', '40', '50', '60', '100', '120', '150', '200', '240', '300', '500']

# here are all the unique characters that occur in this text
vocab_size = len(chars)
output_size = len(outputs)

_inputs = []
_outputs = []

for line in text.splitlines():
    parts = line.split(',')
    subtext = parts[0].upper()
    part_length = len(parts)
    for c in range(len(subtext)):
        _input = [0] * block_size
        _output = [0] * len(outputs)
        for i in range(block_size):
            if c + i < len(subtext):
                _input[i] = chars.index(subtext[c + i])
        _output[int(line.split(',')[part_length - 1])] = 1
        _inputs.append(_input)
        _outputs.append(_output)

# Train and test splits
inputs = torch.tensor(_inputs, dtype=torch.long)
outputs = torch.tensor(_outputs, dtype=torch.float)
n = int(0.9*len(inputs)) # first 90% will be train, rest val
train_data = inputs[:n]
train_outputs = outputs[:n]
val_data = inputs[n:]
val_outputs = outputs[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    data_out = train_outputs if split == 'train' else val_outputs
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i] for i in ix])
    y = torch.stack([data_out[i] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class OLFDataset(Dataset):
    def __init__(self, words, chars, max_len):
        self.txt_path = "/workspaces/OLF-Data/OLFNetworkData.txt"
        self.data = []
        with open('OLFNetworkData.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            for line in text.splitlines():
                self.data.append([line.split(',')[0], line.split(',')[-1]])
        self.class_map = {"0" : 0, "5" : 1, "7" : 2, "11" : 3, "15" : 4, "20" : 5, "30" : 6, "35" : 7, "40" : 8, "50" : 9, "60" : 10, "100" : 11, "120" : 12,
                          "150" : 13, "200" : 14, "240" : 15, "300" : 16, "500" : 17}
    
    def __len__(self):
        return len(self.data)
    
    def contains(self, word):
        return word in self.words
    
    def get_output_length(self):
        return self.max_len + 1
    
    def __getitem__(self, idx):
        data, class_name = self.data[idx]
        class_id = self.class_map[class_name]
        class_id = torch.tensor([class_id])
        x = torch.zeros(self.max_len + 1, dtype=torch.long)
        class_id = self.class_map[]
        x[1:1+len(ix)] = ix


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class XfmrModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, output_size)
        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        logits = torch.sum(x, dim=-2, keepdim=False)

        if targets is None:
            loss = None
        else:
            #B, C = logits.shape
            #logits = logits.view(B, C)
            #targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

def RunTraining():
    for iter in range(max_iters):
        if iter % 500 == 0 or iter == max_iters - 1:
            torch.save(model.state_dict(), path)
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

path = "/workspaces/OLF-Data/OLFNetwork.pt"
model = XfmrModel()
if os.path.isfile(path):
    statedict = torch.load(path)
    model.load_state_dict(statedict)

m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

while True:
    usage = input("Train or Test?")
    if usage == "Test":
        test = ""
        while test != "X":
            test = input("Test your room name")
            test_inputs = []
            for char in range(len(test)):
                _input = [0] * block_size
                for i in range(block_size):
                    if char + i < len(text):
                        if(chars.__contains__(text[c+i])):
                            _input[i] = chars.index(text[c + i])
                test_inputs.append(_input)
            test_input = torch.tensor(test_inputs)
            test_output = model(test_input)
            print(test_output)
    elif usage == "Train":
        RunTraining()