import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=32*32*3, hidden1=1024, hidden2=512, num_classes=10, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden2, num_classes),
        )
    
    def forward(self, x):
        return self.net(x)


class SmallCIFARConvNet(nn.Module):
    def __init__(self, num_classes=10, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 

            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(128*8*8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        return self.classifier(x)
