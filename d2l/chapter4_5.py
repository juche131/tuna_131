import matplotlib
import torch
from d2l import torch as d2l
from torch import nn
import euinyee

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w , true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
train_data = euinyee.synthetic_data(true_w, true_b, n_train)
train_iter = euinyee.load_array(train_data, batch_size)
test_data = euinyee.synthetic_data(true_w, true_b, n_test)
test_iter = euinyee.load_array(test_data, batch_size, is_train=False)

def init_params():
    """Initialize model parameters."""
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    """L2 regularization penalty."""
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b= init_params()
    net, loss = lambda X: euinyee.linreg(X, w, b), euinyee.squared_loss
    num_epochs, lr = 100, 0.03
    animator = euinyee.Animator(xlabel='epoch', ylabel='loss',
                                yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            euinyee.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,(
                euinyee.evaluate_loss(net, train_iter, loss),
                euinyee.evaluate_loss(net, test_iter, loss)
            ))
    print("w的L2范数：", torch.norm(w).item())
        
train(lambd=3)  # No regularization