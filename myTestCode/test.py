import torch
# CUDA TEST
print(torch.cuda.is_available())
x = torch.Tensor([1,0])
xx = x.cuda()
print(xx)
print("======")
# CUDNN TEST
from torch.backends import cudnn
print(cudnn.is_acceptable(xx))

