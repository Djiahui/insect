import torch
import torch.nn as nn


t = torch.rand(5,1,11,11)
net = nn.Conv2d(1,1,kernel_size=2,stride=2)
net2 = nn.Conv2d(1,1,kernel_size=2,stride=2)
net3 = nn.Conv2d(1,1,kernel_size=2,stride=1)
out = net(t)
print(out.size())
out = net2(out)
print(out.size())
out = net3(out)
print(out.size())

exit(0)

