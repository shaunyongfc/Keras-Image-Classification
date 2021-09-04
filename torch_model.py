import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassTorch(nn.Module):
    def __init__(self):
        """
        Create the neural network model.
        """
        super(ImageClassTorch, self).__init__()
        self.model = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # Second convolutional layer
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3),
            nn.Dropout(0.25),
            # Third convolutional layer
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(5),
            nn.Dropout(0.25),
            # First fully connected layer
            nn.Flatten(),
            nn.Linear(32*5*5, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            # Final fully connected layer
            nn.Linear(100, 6)
        )
        
    def forward(self, x):
        """
        Compute output tensors from input x.
        """
        return self.model(x)
