import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGFCN(nn.Module):
    def __init__(self, num_classes=2, unet_classes=2, in_channels=12, unet_initial_filter_size=32, kernel_size=5):
        super(ECGFCN, self).__init__()
        self.fcn = FCN8s1D(num_classes=unet_classes, in_channels=in_channels, kernel_size=kernel_size)

        self.classifier = Classifier(num_classes, in_channels)

    def forward(self, x):
        result = self.fcn(x)
        softmax = F.softmax(result, dim=1)
        # atten
        atten = softmax[:, :1]
        atten1 = softmax[:, -1:]
        if atten.shape != x.shape:
            atten = atten.expand_as(x)
            atten1 = atten1.expand_as(x)

        x = torch.mul(x, atten)
        x1 = torch.mul(x, atten1)

        y = self.classifier(x)
        y1 = self.classifier(x1)
        return atten, y, y1


class FCN8s1D(nn.Module):
    def __init__(self, num_classes=2, in_channels=12, kernel_size=5):
        super(FCN8s1D, self).__init__()
        self.model_name = 'FCN8s'
        self.n_classes = num_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv1d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv1d(4096, self.n_classes, 1),
        )

        self.score_pool2 = nn.Conv1d(128, self.n_classes, 1)
        self.score_pool3 = nn.Conv1d(256, self.n_classes, 1)

        self.upscore = nn.ConvTranspose1d(self.n_classes, self.n_classes, 16, stride=4)
        self.upscore3 = nn.ConvTranspose1d(self.n_classes, self.n_classes, 4, stride=2)
        self.upscore4 = nn.ConvTranspose1d(self.n_classes, self.n_classes, 4, stride=2)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2) # c = 256, 1/8
        conv4 = self.conv_block4(conv3) # c = 512, 1/16
        score4 = self.classifier(conv4)

        upscore4 = self.upscore4(score4)
        score3 = self.score_pool3(conv3)
        score3 = score3[:, :, 5:5 + upscore4.size()[2]].contiguous()
        score3 += upscore4

        upscore3 = self.upscore3(score3)
        score2 = self.score_pool2(conv2)
        score2 = score2[:, :, 9:9 + upscore3.size()[2]].contiguous()
        score2 += upscore3

        out = self.upscore(score2)
        out = out[:, :, 31:31 + x.size()[2]].contiguous()
        return out


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

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

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

        x = torch.flatten(self.gap(x), start_dim=1)
        x = self.fc1(x)
        y = self.fc2(x)

        return y


