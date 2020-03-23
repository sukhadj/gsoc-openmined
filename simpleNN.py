import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, NUM_CLASS=10):
        super(SimpleNN, self).__init__()
        
        # Quad layer
        self.proj1 = nn.Linear(784, 40)
        self.diag1 = nn.Linear(40, NUM_CLASS, bias=False) # why bias false?
        
        # Layer that substitutes argmax function
        self.lin1 = nn.Linear(NUM_CLASS, 32)
        self.lin2 = nn.Linear(32, NUM_CLASS)
        
    def forward(self, x):
        # quad layer
        x = x.view(-1, 784) # flatten the image
        x = self.proj1(x)
        x = x * x # quadratic function
        x = self.diag1(x)
        
        # prediction 
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        out = self.lin2(x)
        
        return F.log_softmax(out, dim=1)