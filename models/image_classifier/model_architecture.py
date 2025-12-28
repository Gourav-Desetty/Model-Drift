from pathlib import Path
import torch 
from torch import nn
from torchvision import models

class LeukemiaCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True) :
        super().__init__()

        self.model = models.densenet121(weights=None)
        # self.model = models.efficientnet_b0(pretrained=pretrained)

        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),  # Reduced from 0.6
            nn.Linear(num_features, 256),  # Increased capacity
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Added batch norm
            nn.Dropout(0.3),  # Reduced from 0.4
            nn.Linear(256, num_classes)
        )


        # num_features = self.model.classifier[1].in_features
        # self.model.classifier = nn.Sequential(
        #     nn.Dropout(0.5),  #0.6
        #     nn.Linear(in_features=num_features, out_features=128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),  #0.4
        #     nn.Linear(128, num_classes)
        # )

    def forward(self, X):
        return self.model(X)