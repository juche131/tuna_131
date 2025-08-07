import matplotlib
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from d2l import torch as d2l
import euinyee

d2l.use_svg_display()

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True)

x,y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
euinyee.show_images(x.reshape((18, 28, 28)), 2, 9, titles=euinyee.get_fasion_mnist_labels(y))

batch_size = 256
def get_dataloader_workers():
    """Use 4 processes to read the data."""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

timer = euinyee.Timer()
for X, y in train_iter:
    continue
print(f'{len(train_iter.dataset)} examples, {timer.stop():.5f} sec')

train_iter, test_iter = euinyee.load_data_fasion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break