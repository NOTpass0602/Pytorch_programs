import torch
from torch import nn
from tqdm import tqdm
from Net import Model
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
#实例化
D=Dataset()
train_data_size=len(D.train_data())
test_data_size=len(D.test_data())
print("数据集的长度为：{}".format(train_data_size))
class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Model().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters())
        self.train_loader=DataLoader(D.train_data(),batch_size=60,shuffle=True)
        self.test_loader = DataLoader(D.test_data(), batch_size=60, shuffle=False)

    def evaluate(self, test_data, net):
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for inputs, labels in test_data:
                #这里也要确保图片和标签在同一设备
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = net.forward(inputs.view(-1, 28 * 28))
                #print("",outputs)
                #outputs是一个维度为10的横向向量，取这10个向量中最大的数为第i个，outputs.argmax(1)的值即为i，用于与labels的值对应，.sum加和求总准确的量
                n_correct = (outputs.argmax(1) == labels).sum()
                n_total = n_total + n_correct
                #for i, output in enumerate(outputs):
                    #if torch.argmax(output) == labels[i]:
                        #n_correct += 1
                    #n_total += 1
        return n_total / test_data_size

    def train(self,stop_value):
        epoch=1
        while True:
            #在这里想使用数据集中的数据必须要使用dataloader处理后的数据，如用self.test_loader代替D.test_data（处理前）
            print("initial accuracy:{}".format(self.evaluate(self.test_loader, self.net)))
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch}/{stop_value}",
                                       ascii=True, total=len(self.train_loader)):
                #确保数据也在正确的设备上
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                #print("data shape:",inputs.shape)
                #print("label shape:",labels.shape)
                #print("",labels)
                #计算前向传播结果
                out=self.net.forward(inputs.view(-1,28*28))
                #计算out和标签之间的损失
                loss = F.nll_loss(out,labels)
                #梯度清零
                self.opt.zero_grad()
                #用backward计算梯度
                loss.backward()
                #用opt.step更新参数
                self.opt.step()
            print(f"\nEpoch: {epoch}/{stop_value}, Loss: {loss}")
            print("accuracy:",self.evaluate(self.test_loader,self.net))
            if epoch >=stop_value:
                break
            epoch+=1

if __name__ == '__main__':
    t=Trainer()
    t.train(3)