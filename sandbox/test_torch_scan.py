import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

from tqdm import tqdm

torch.manual_seed(0)

# Parameters
B, L, D_in, D_h, D_out = 32, 512, 64, 128, 10
x = torch.randn(B, L, D_in).cuda()
y = torch.randint(0, D_out, (B,)).cuda()

# RNN cell
class RNNCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.i2h = nn.Linear(D_in, D_h)
        self.h2h = nn.Linear(D_h, D_h)

    def forward(self, x_t, h_prev):
        return torch.tanh(self.i2h(x_t) + self.h2h(h_prev))

# Classifier
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(D_h, D_out)

    def forward(self, h):
        return self.out(h)

# Naive RNN model
class NaiveRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = RNNCell()
        self.head = Head()

    def forward(self, x):
        h = torch.zeros(x.size(0), D_h, device=x.device)
        for t in range(x.size(1)):
            h = self.cell(x[:, t], h)
        return self.head(h)

# RNN using torch.scan
class ScanRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = RNNCell()
        self.head = Head()

    def forward(self, x):
        def step(h, x_t):
            h = self.cell(x_t, h)
            return h, h
        h0 = torch.zeros(x.size(0), D_h, device=x.device)
        _, h_seq = torch.scan(step, x.transpose(0, 1), h0)
        return self.head(h_seq[-1])

# Train step
def train_step(model, optimizer, x, y):
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Models and optimizers
naive_model = NaiveRNN().cuda()
opt1 = torch.optim.Adam(naive_model.parameters(), lr=1e-3)


# Warm-up
print("Warm-up for naive model")
for _ in range(5):
    train_step(naive_model, opt1, x, y)

# Timing
start = time.time()
for _ in tqdm(range(100)):
    train_step(naive_model, opt1, x, y)
time_naive = time.time() - start


print("Warm-up for torch compile model")

naive_model_compiled = torch.compile(NaiveRNN().cuda())

for _ in range(5):
    train_step(naive_model_compiled, opt1, x, y)

start = time.time()
for _ in tqdm(range(100)):
    train_step(naive_model_compiled, opt1, x, y)
time_naive_compiled = time.time() - start


print(f"Naive loop time: {time_naive:.3f}s")
print(f"Naive compiled time: {time_naive_compiled:.3f}s")
print(f"Speedup: {time_naive / time_naive_compiled:.2f}x")
