import time
import matplotlib
import torch
from d2l import torch as d2l
import random
from torch.utils import data
from torchvision import transforms
import torchvision
from IPython import display
import matplotlib.pyplot as plt


# === 0 通用模块 ===

matplotlib.use('Qt5Agg')  # 设置matplotlib的后端为Qt5Agg

# 计时器类
class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start()

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        print("Timer started.")

    def stop(self):
        """Stop the timer and return the elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        print(f"Timer stopped. Elapsed time: {elapsed_time:.5f} seconds.")
        return elapsed_time

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        print("Timer reset.")
        
# 在n个变量上累加的Accumulator类
class Accumulator:
    """Accumulate sums of n variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        """Add values to the accumulator."""
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """Reset the accumulator."""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """Get the value at index idx."""
        return self.data[idx]
        
        
# 在动画中绘制图表的实用程序类Animator
class Animator:
    """Plot data in an animated way."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'),
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
            
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        
    def add(self, x, y):
        """Add data to the plot."""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            self.X[i].append(a)
            self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        
# === 1 数据模块 ===

# 生成合成数据的函数
def synthetic_data(w, b, num_examples):
    """Generate synthetic data for linear regression."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # Add some noise
    return X, y.reshape((-1, 1))

# Fashion-MNIST标签转换函数
def get_fasion_mnist_labels(labels):
    """Return text labels for Fashion-MNIST."""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]

def get_dataloader_workers():
    """Use 4 processes to read the data."""
    return 4

# 下载Fashion-MNIST数据集并返回数据迭代器
def load_data_fasion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and return data iterators."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
    
# 显示图像的函数
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Display a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# 数据加载器
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# === 2 模型模块 ===

# 线性回归模型
def linreg(X, w, b):
    """Linear regression model."""
    return torch.matmul(X, w) + b

# === 3 优化模块 ===

# 损失函数
def squared_loss(y_hat, y):
    """Squared loss function."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 优化算法
def sgd(params, lr, batch_size):
    """Stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
# 计算预测正确的数量
def accuracy(y_hat, y):
    """Calculate the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 计算在指定数据集上模型的精度
def evaluate_accuracy(net, data_iter):
    """Evaluate the accuracy of the model on the dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = d2l.Accumulator(2)  # Correct predictions, total predictions
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]  # Return accuracy

# 计算在指定数据集上的损失
def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of the model on the dataset."""
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(float(l.sum()), l.numel())
    return metric[0] / metric[1]  # Return average loss

# === 4 迭代模块 ===

# 训练一个epoch
def train_epoch_ch3(net, train_iter, loss, updater):
    """Train a single epoch of the model."""
    if isinstance(net, torch.nn.Module):
        net.train()  # Set the model to training mode
    metric = Accumulator(3)  # Sum of training loss, number of examples, number of correct predictions
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Use PyTorch's built-in optimizer
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]  # Return average loss and accuracy

# 训练模型
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train the model."""
    animator = Animator(xlabel='epoch', xlim=[1,num_epochs], ylim=[0.3,0.9], ylabel='loss',
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics+(test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss<0.5, train_loss
    assert train_acc<=1 and train_acc>0.7, train_loss
    assert test_acc<=1 and test_acc >0.7, test_acc
    plt.show()
