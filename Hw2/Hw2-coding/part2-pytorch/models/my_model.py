import torch
import torch.nn as nn


class MyModel(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        # Conv Block 1: 2 conv layers + batchnorm + ReLU + maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: 2 conv layers + batchnorm + skip connection + ReLU + maxpool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv_skip2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)  # 1x1 conv for skip connection
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv_skip3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        #Conv Block 1
        x = self.relu(self.bn1(self.conv1(x)))  # (N, 64, 32, 32)
        x = self.relu(self.bn2(self.conv2(x)))  # (N, 64, 32, 32)
        x = self.pool1(x)                       # (N, 64, 16, 16)
        
        #Conv Block 2 with skip connection
        identity = x                            # Save input for skip connection
        x = self.relu(self.bn3(self.conv3(x)))  # (N, 128, 16, 16)
        x = self.bn4(self.conv4(x))             # (N, 128, 16, 16)
        identity = self.conv_skip2(identity)     # (N, 128, 16, 16)
        x = self.relu(x + identity)             # Skip connection: add input
        x = self.pool2(x)                       # (N, 128, 8, 8)
        

        # Conv Block 3
        identity = x                             # (N, 256, 8, 8)
        x = self.relu(self.bn5(self.conv5(x)))   # (N, 256, 8, 8)
        x = self.bn6(self.conv6(x))              # (N, 256, 8, 8)
        identity = self.conv_skip3(identity)     # (N, 256, 8, 8)    
        x = self.relu(x + identity)              # Skip connection
        x = self.pool3(x)                        # (N, 256, 4, 4)


        # Fully connected
        x = x.view(x.size(0), -1)               # (N, 4096)
        x = self.relu(self.fc1(x))              # (N, 512)
        x = self.dropout(x)                     # (N, 512)
        outs = self.fc2(x)                      # (N, 10)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs