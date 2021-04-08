import torch.nn as nn
import torch.nn.functional as F
import torch


class RNNClassifier(nn.Module):
    def __init__(self):
        super(RNNClassifier, self).__init__()
        self.conv = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5, stride=1)
        # self.bn = nn.BatchNorm1d(num_features=32)

        # self.conv_pad = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        #                               nn.BatchNorm1d(num_features=32))

        self.conv_pad = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.drop_50 = nn.Dropout(p=0.5)

        self.conv1_1 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)

        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2)

        self.rnn = nn.LSTM(32, 64, 2, dropout=0.3)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 2)
        self.drop_20 = nn.Dropout(p=0.2)

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

        x = x.permute(2, 0, 1)
        output, (hn, cn) = self.rnn(x)

        hn = hn.transpose(0, 1)
        hn = torch.flatten(hn, start_dim=1)
        y = self.fc1(hn)
        y = self.drop_20(y)
        y = self.fc2(y)

        return y



class Classifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=12):
        super(Classifier, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.BatchNorm1d(num_features=32),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(num_features=64),
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(num_features=64),
        )

        self.block5 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, stride=1),
            nn.BatchNorm1d(num_features=128),
        )

        self.drop_60 = nn.Dropout(p=0.6)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        # self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(128, num_classes)
        # self.fc2 = nn.Linear(256, num_classes)

        # self.rnn = nn.LSTM(128, 64, 2, dropout=0.3, bidirectional=True)

    def forward(self, x):
        x = self.block1(x)
        x = F.relu(x)
        x = self.block2(x)
        x = F.relu(x)
        x = self.block3(x)
        x = F.relu(x)
        x = self.block4(x)
        x = F.relu(x)
        x = self.block5(x)
        x = F.relu(x)

        x = x.permute(2, 0, 1)
        output, (hn, cn) = self.rnn(x)

        hn = hn.transpose(0, 1)
        hn = torch.flatten(hn, start_dim=1)
        y = F.relu(self.fc1(hn))
        y = self.drop_60(y)

        y = F.relu(self.fc2(x))

        return y


