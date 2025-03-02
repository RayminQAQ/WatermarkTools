import torch.nn as nn
import torch.nn.functional as F

class SimpleAE(nn.Module):
    def __init__(self, sample, in_dim, hid_dim, out_dim):
        super(SimpleAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(hid_dim, out_dim)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
if __name__ == "__main__":
    import torch
    image = torch.zeros(64, 64)
    
    model = SimpleAE()
    
    # return the same shape image
    image = model.forward(image)
    
    print(image.shape)