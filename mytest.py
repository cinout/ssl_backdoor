import torch

bs = 20
c = 3
h = 10
w = 10
mask = torch.randn(bs, 3, h, w)
result = torch.norm(mask, p=1, dim=[1, 2, 3])

print(result)
print(result.shape)
