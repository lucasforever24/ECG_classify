import torch.nn as nn
import torch.nn.functional as F
import torch


class AnomalyClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AnomalyClassifier, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1)
        # self.bn = nn.BatchNorm1d(num_features=32)

        # self.conv_pad = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        #                               nn.BatchNorm1d(num_features=32))

        self.conv_pad = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.drop_50 = nn.Dropout(p=0.5)

        # self.conv1_1 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)

        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.dense1 = nn.Linear(32, 32)
        self.dense2 = nn.Linear(32, 32)

        self.dense_final = nn.Linear(32, num_classes)

    def forward(self, x):
        residual = self.conv(x)

        # block1
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        # x += residual
        x = F.relu(x)
        residual = self.maxpool(x)  # [512 32 90]

        # block2
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        # x += residual
        x = F.relu(x)
        residual = self.maxpool(x)  # [512 32 43]

        # block3
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        # x += residual
        x = F.relu(x)
        residual = self.maxpool(x)  # [512 32 20]

        # block4
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        # x += residual
        x = F.relu(x)
        x = self.maxpool(x)  # [512 32 8]
        # x = self.conv1_1(x)
        x = self.gap(x)

        # MLP
        x = torch.flatten(x, start_dim=1)  # Reshape (current_dim, 32*2)
        x = F.relu(self.dense1(x))
        # x = self.drop_50(x)
        # x = F.relu(self.dense2(x))
        x = self.dense_final(x)

        return x

