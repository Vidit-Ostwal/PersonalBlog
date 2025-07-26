import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# Config
d_model = 3
t = 3  # Number of tokens
vmin, vmax = -500, 500  # Fixed color scale
torch.manual_seed(43)

# Random integer inputs and weights
x = torch.randint(-5, 6, (t, d_model)).float()
W_q = torch.randint(-5, 6, (d_model, d_model)).float()
W_k = torch.randint(-5, 6, (d_model, d_model)).float()
W_v = torch.randint(-5, 6, (d_model, d_model)).float()

# Compute Q, K, V and Attention outputs
Xs, Qs, Ks, Vs = [], [], [], []
Scores, AttnWeights, Outputs = [], [], []

for step in range(1, t + 1):
    x_now = x[:step]
    Q = x_now @ W_q
    K = x_now @ W_k
    V = x_now @ W_v

    Xs.append(x_now.detach())  # ✅ Fixed typo here
    Qs.append(Q.detach())
    Ks.append(K.detach())
    Vs.append(V.detach())

    attn_scores = Q @ K.T / math.sqrt(d_model)
    attn_weights = F.softmax(attn_scores, dim=-1)
    output = attn_weights @ V

    Scores.append(attn_scores.detach())
    AttnWeights.append(attn_weights.detach())
    Outputs.append(output.detach())

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
    plot_matrix(axes[2, col], Ks[col], f"K @ Step {col+1}")
    plot_matrix(axes[3, col], Vs[col], f"V @ Step {col+1}")
    plot_matrix(axes[4, col], Outputs[col], f"Attention Output @ Step {col+1}")

# Layout tweaks
plt.subplots_adjust(
    left=0.06, right=0.96,
    top=0.95, bottom=0.05,
    wspace=0.5, hspace=0.6
)

plt.show()
