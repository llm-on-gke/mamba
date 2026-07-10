import torch
import torch.nn.functional as F

C = 2
d_conv = 4
xBC = torch.randn(1, 5, C)
conv1d = torch.nn.Conv1d(C, C, kernel_size=d_conv, padding=d_conv-1, groups=C)

# 1. nn.Conv1d
xBC_t = xBC.transpose(1, 2)
y1 = conv1d(xBC_t).transpose(1, 2)[:, :5, :]

# 2. Manual
w = conv1d.weight.squeeze(1)
b = conv1d.bias
out_conv = torch.zeros_like(xBC)
for i in range(d_conv):
    shift = d_conv - 1 - i
    if shift == 0:
        shifted_x = xBC
    else:
        shifted_x = F.pad(xBC[:, :-shift, :], (0, 0, shift, 0))
    out_conv += shifted_x * w[:, i]
y2 = out_conv + b

print(torch.allclose(y1, y2, atol=1e-5))
