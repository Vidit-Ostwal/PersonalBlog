import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

# Parameters
d_model = 512     # Embedding dimension
t = 200           # Sequence length
x = torch.randn(t, d_model)  # Random input sequence

# Random projection matrices
W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)

# Store timings
timings_no_cache = []
timings_kv_cache = []

# Initialize KV cache
K_cache = []
V_cache = []

# Simulate decoding without KV cache
for step in range(1, t + 1):
    x_till_now = x[:step]

    start = time.time()

    # Full recomputation
    Q = x_till_now @ W_q
    K = x_till_now @ W_k
    V = x_till_now @ W_v

    scores = Q @ K.T / (d_model ** 0.5)
    weights = torch.nn.functional.softmax(scores, dim=-1)
    output = weights @ V

    end = time.time()
    elapsed_ms = (end - start) * 1000  # milliseconds
    timings_no_cache.append(elapsed_ms)


for step in range(t):
    x_t = x[step : step + 1]  # (1, d_model)

    start = time.time()

    Q = x_t @ W_q             # (1, d_model)
    K_t = x_t @ W_k
    V_t = x_t @ W_v

    K_cache.append(K_t)       # append (1, d_model)
    V_cache.append(V_t)

    K_all = torch.cat(K_cache, dim=0)  # (step+1, d_model)
    V_all = torch.cat(V_cache, dim=0)  # (step+1, d_model)

    scores = Q @ K_all.T / (d_model ** 0.5)   # (1, step+1)
    weights = F.softmax(scores, dim=-1)
    output = weights @ V_all                 # (1, d_model)

    end = time.time()
    elapsed_ms = (end - start) * 1000
    timings_kv_cache.append(elapsed_ms)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, t + 1), timings_kv_cache, label='With KV Cache', color='blue')
plt.plot(range(1, t + 1), timings_no_cache, label='No KV Cache', color='red')
plt.xlabel('Decoding Step')
plt.ylabel('Time (ms)')
plt.title('Time Taken per Decoding Step (With KV Cache)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

