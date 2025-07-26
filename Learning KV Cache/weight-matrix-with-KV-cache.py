import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# Config
d_model = 3                 # Embedding dimension
t = 3                       # Number of tokens (timesteps)
vmin, vmax = -500, 500      # For consistent color scale in plots
torch.manual_seed(43)

# Random integer inputs and weights
x = torch.randint(-5, 6, (t, d_model)).float()           # x: (t=3, d_model=3)
W_q = torch.randint(-5, 6, (d_model, d_model)).float()   # W_q: (3, 3)
W_k = torch.randint(-5, 6, (d_model, d_model)).float()   # W_k: (3, 3)
W_v = torch.randint(-5, 6, (d_model, d_model)).float()   # W_v: (3, 3)

# Buffers for caching keys and values
cached_K = []  # List of tensors, each (1, d_model)
cached_V = []  # List of tensors, each (1, d_model)

# Store history for visualization
Xs, Qs, Ks, Vs, Outputs = [], [], [], [], []

for step in range(1, t + 1):
    x_now = x[step - 1 : step]         # x_now: (1, 3), current token only

    Q = x_now @ W_q                    # Q: (1, 3)
    K = x_now @ W_k                    # K: (1, 3)
    V = x_now @ W_v                    # V: (1, 3)

    # Cache K and V for future steps
    cached_K.append(K)                # cached_K is a list of (1, 3)
    cached_V.append(V)                # cached_V is a list of (1, 3)

    K_cat = torch.cat(cached_K, dim=0)  # K_cat: (step, 3)
    V_cat = torch.cat(cached_V, dim=0)  # V_cat: (step, 3)

    # Compute attention scores
    attn_scores = Q @ K_cat.T           # (1, 3) @ (3, step).T -> (1, step)
    attn_scores = attn_scores / math.sqrt(d_model)  # scale: (1, step)

    attn_weights = F.softmax(attn_scores, dim=-1)    # attn_weights: (1, step)

    output = attn_weights @ V_cat       # (1, step) @ (step, 3) -> (1, 3)

    # Store for visualization
    Xs.append(x_now.detach())          # (1, 3)
    Qs.append(Q.detach())              # (1, 3)
    Ks.append(K_cat.detach())          # (step, 3)
    Vs.append(V_cat.detach())          # (step, 3)
    Outputs.append(output.detach())    # (1, 3)


# Plot helper
def plot_matrix(ax, matrix, title, cmap="Blues"):
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=10)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xlabel("Dim", labelpad=8)
    ax.set_ylabel("Token", labelpad=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j].item()
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=11, fontfamily="monospace")
    return im

# ✅ Plot with 7 rows now
fig, axes = plt.subplots(5, t, figsize=(4 * t, 21))

for col in range(t):
    plot_matrix(axes[0, col], Xs[col], f"X @ Step {col+1}")
    plot_matrix(axes[1, col], Qs[col], f"Q @ Step {col+1}")
    plot_matrix(axes[2, col], Ks[col], f"K Cache @ Step {col+1}")
    plot_matrix(axes[3, col], Vs[col], f"V Cache @ Step {col+1}")
    plot_matrix(axes[4, col], Outputs[col], f"Attention Output @ Step {col+1}")

# Layout tweaks
plt.subplots_adjust(
    left=0.06, right=0.96,
    top=0.95, bottom=0.05,
    wspace=0.5, hspace=0.6
)

plt.show()
