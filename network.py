from abc import ABC, abstractmethod
from math import floor
import torch.nn as nn
import torch.nn.functional as F
import torch

class AbstractNetwork(nn.Module, ABC):
    
    @abstractmethod
    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        pass

class SimpleNetwork(AbstractNetwork):
    
    _INTER_NUM = 86

    def __init__(self, img_dim: int, out_num: int):
        super().__init__()
        self.fc1 = nn.Linear(img_dim * img_dim * 3, self._INTER_NUM)
        self.fc2 = nn.Linear(self._INTER_NUM, self._INTER_NUM)
        self.fc3 = nn.Linear(self._INTER_NUM, self._INTER_NUM)
        self.fc4 = nn.Linear(self._INTER_NUM, out_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x= self.fc4(x)
        return F.log_softmax(x, dim=1)

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, self.fc1.in_features)

class Network(AbstractNetwork):
    
    def __init__(self, img_dim: int, out_num: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        dim_trans = self._after_pool(img_dim, 2)
        #print(dim_trans2)

        self.fc1 = nn.Linear(16 * dim_trans * dim_trans, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_num)

    def _after_pool(self, dim: int, count: int = None):
        padding = self.pool.padding
        dilation = self.pool.dilation
        k_size = self.pool.kernel_size
        stride = self.pool.stride
        
        numerator = dim + 2 * padding - dilation * (k_size - 1) - 1
        y = floor(numerator / stride + 1) - 2

        count = count or 1
        for _ in range(count - 1):
            numerator = y + 2 * padding - dilation * (k_size - 1) - 1
            y = floor(numerator / stride + 1) - 2

        return y

    def forward(self, x):
        #print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        return x
