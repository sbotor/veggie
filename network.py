import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self, in_num: int, out_num: int, inter_num: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_num, inter_num)
        self.fc2 = nn.Linear(inter_num, inter_num)
        self.fc3 = nn.Linear(inter_num, inter_num)
        self.fc4 = nn.Linear(inter_num, out_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x= self.fc4(x)
        return F.log_softmax(x, dim=1)
