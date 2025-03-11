import torch.nn as nn
from util.cnn_parameters import KERNEL_SIZE_CONV, PADDING, POOL_SIZE, MAXPOOL_STRIDE

# For lrp method

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=KERNEL_SIZE_CONV, padding=PADDING),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=POOL_SIZE, stride=MAXPOOL_STRIDE),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE_CONV, padding=PADDING),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=POOL_SIZE, stride=MAXPOOL_STRIDE),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE_CONV, padding=PADDING),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=POOL_SIZE, stride=MAXPOOL_STRIDE),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=KERNEL_SIZE_CONV, padding=PADDING),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=POOL_SIZE, stride=MAXPOOL_STRIDE),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, kernel_size=KERNEL_SIZE_CONV, padding=PADDING),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=POOL_SIZE, stride=MAXPOOL_STRIDE),

            nn.Flatten(),

            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
