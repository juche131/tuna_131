import torch
from IPython import display
from d2l import torch as d2l
import euinyee

batch_size = 256
train_iter, test_iter = euinyee.load_data_fasion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

X = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
# print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))

def softmax(X):
    """Compute the softmax for each row of the input X."""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这里应用了广播机制

X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
# print(X_prob, X_prob.sum(1))

def net(X):
    """The model."""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6],
                     [0.3, 0.2, 0.5]])
# print(y_hat[[0, 1], y])

def cross_entropy(y_hat, y):
    """Cross-entropy loss."""
    return -torch.log(y_hat[range(len(y_hat)), y])
# print(cross_entropy(y_hat, y))

lr = 0.1
def updater(batch_size):
    return euinyee.sgd([W,b], lr, batch_size)

num_epochs = 10
euinyee.train_ch3(net, train_iter, test_iter,cross_entropy,num_epochs,updater)