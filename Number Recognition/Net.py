import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):#M要大写
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(28*28,256)
        self.l2=nn.Linear(256,128)
        self.l5 = nn.Linear(128, 64)
        self.l3=nn.Linear(64,32)
        self.l4 = nn.Linear(32, 10)
        
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x = F.relu(self.l5(x))
        x=F.relu(self.l3(x))
        x=F.log_softmax(self.l4(x),1)
        return x

