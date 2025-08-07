import torch
from torch import nn
import euinyee

batch_size = 256
train_iter, test_iter = euinyee.load_data_fasion_mnist(batch_size)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
euinyee.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)