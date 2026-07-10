import torch
import torch.nn.functional as F

C = 2
d_conv = 4
L = 5
B = 1
xBC = torch.randn(B, L, C)
conv1d = torch.nn.Conv1d(C, C, kernel_size=d_conv, padding=d_conv-1, groups=C)

# 1. nn.Conv1d
xBC_t = xBC.transpose(1, 2)
y1 = conv1d(xBC_t).transpose(1, 2)[:, :L, :]

# 2. Banded matrix
w = conv1d.weight.squeeze(1) # (C, d_conv)
b = conv1d.bias # (C,)

idx = torch.arange(L).unsqueeze(1) - torch.arange(L).unsqueeze(0)
valid = (idx >= 0) & (idx < d_conv)
idx_clamped = torch.where(valid, idx, d_conv)

w_padded = torch.cat([w, torch.zeros(C, 1)], dim=1) # (C, d_conv + 1)
# To get W[c, i, j] = w_padded[c, idx_clamped[i, j]]
# PyTorch advanced indexing:
# W = w_padded[:, idx_clamped] # This is (C, L, L)
W = w_padded[:, idx_clamped]

# The convolution sum is: y[t] = \sum_k x[k] * W[t, k]
# Wait, y_c[t] = \sum_k W_c[t, k] * xBC_{b, k, c}
# out_conv[b, t, c] = \sum_k W[c, t, k] * xBC[b, k, c]
out_conv = torch.einsum('c t k, b k c -> b t c', W, xBC)
y2 = out_conv + b

print(torch.allclose(y1, y2, atol=1e-5))
