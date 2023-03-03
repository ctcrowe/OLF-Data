import torch
import torch.nn as nn
import time
import os
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
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
    def __init__(self, lines):
        #self.txt_path = "/workspaces/OLF-Data/OLFNetworkData.txt"
        self.data = []
        self.chars = [' ', ',', '_', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.max_len = 128
        #with open('OLFNetworkData.txt', 'r', encoding='utf-8') as f:
            #text = f.read()
        for line in lines # text.splitlines():
            line = line.strip().upper()
            classification = line.split(',')[-1]
            line = [c if self.chars.count(c) > 0 else '' for c in line]
            self.data.append([line, classification])
        self.class_map = {"0" : 0, "5" : 1, "7" : 2, "11" : 3, "15" : 4, "20" : 5, "30" : 6, "35" : 7, "40" : 8, "50" : 9, "60" : 10, "100" : 11, "120" : 12,
                          "150" : 13, "200" : 14, "240" : 15, "300" : 16, "500" : 17}
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, class_name = self.data[idx]
        class_id = self.class_map[class_name]
        class_id = torch.tensor([class_id])
        ix = torch.tensor([self.stoi[w] for w in data], dtype = torch.long)
        x = torch.zeros(self.max_len, dtype=torch.long)
        x[1:1+len(ix)] = ix
        return x, class_id
    
def create_datasets(input_file):
    with open(input_file, 'r') as f:
        data = f.read()
    inputs = data.splitlines()

    test_set_size = min(1000, int(len(data)) * 0.1)
    rp = torch.randperm(len(data)).tolist()
    train_words = [data[i] for i in rp[:-test_set_size]]
    test_words = [data[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    train_dataset = OLFDataset(train_words)
    test_dataset = OLFDataset(test_words)
    return train_dataset, test_dataset

class InfiniteDataLoader:
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler()
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)
    
    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

def evaluate(model, dataset, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss

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

txt_path = "/workspaces/OLF-Data/OLFNetworkData.txt"

path = "/workspaces/OLF-Data/OLFNetwork.pt"
model = XfmrModel()
if os.path.isfile(path):
    statedict = torch.load(path)
    model.load_state_dict(statedict)

m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

def RunTraining():
    train_dataset, test_dataset = create_datasets(txt_path)
    batch_loader = InfiniteDataLoader(train_dataset)

    best_loss = None
    step = 0

    while True:
        t0 = time.time()
        batch = batch_loader.next()
        batch = [t.to(device) for t in batch]
        X, Y = batch

        logits, loss = model(X, Y)

        model.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        if step < 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=10, max_batches=10)
            test_loss = evaluate(model, test_dataset, batch_size=10, max_batches=10)
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                print(f"test loss {test_loss} is the best so far, saving model to {path}")
                torch.save(model.state_dict(), path)
                best_loss = test_loss
            
            
        #if step > 0 and step % 200 == 0:
        #    print_samples(num=10)

        step+=1

while True:
    usage = input("Train or Test?")
    if usage == "Test":
        test = ""
        while test != "X":
            text = input("Test your room name")
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