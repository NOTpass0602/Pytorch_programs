import torchvision
class Dataset:

    def __init__(self):
        #图像的预处理：转换为张量
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    def train_data(self):
        train_set = torchvision.datasets.MNIST(root="./dataset", train=True, transform=self.trans, download=True)
        return train_set
    def test_data(self):
        test_set = torchvision.datasets.MNIST(root="./dataset", train=False, transform=self.trans, download=True)
        return test_set
