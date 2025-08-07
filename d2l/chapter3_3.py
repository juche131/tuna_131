import numpy as np
import torch
from d2l import torch as d2l
from torch.utils import data
import euinyee

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = euinyee.load_array((features, labels), batch_size)
# print(next(iter(data_iter)))

from torch import nn

net = nn.Sequential(
    nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epoches = 3
for epoch in range(num_epoches):
    for X,y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：',true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：',true_b - b)