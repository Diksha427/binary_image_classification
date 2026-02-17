import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        self.pool = nn.MaxPool2d(2,2)
        
        # We will initialize fc1 later dynamically
        self.fc1 = None
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        # Initialize fc1 dynamically
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
