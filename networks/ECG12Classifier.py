import torch.nn as nn
import torch.nn.functional as F
import torch

class ECG12Classifier(nn.Module):
    def __init__(self, input_channel, initial_channel, num_classes=2):
        super(ECG12Classifier, self).__init__()

        self.encoder_1 = LeadEncoder(initial_channel=initial_channel)
        self.encoder_2 = LeadEncoder(initial_channel=initial_channel)
        self.encoder_3 = LeadEncoder(initial_channel=initial_channel)
        self.encoder_4 = LeadEncoder(initial_channel=initial_channel)
        self.encoder_5 = LeadEncoder(initial_channel=initial_channel)
        self.encoder_6 = LeadEncoder(initial_channel=initial_channel)
        self.encoder_7 = LeadEncoder(initial_channel=initial_channel)
        self.encoder_8 = LeadEncoder(initial_channel=initial_channel)
        self.encoder_9 = LeadEncoder(initial_channel=initial_channel)
        self.encoder_10 = LeadEncoder(initial_channel=initial_channel)
        self.encoder_11 = LeadEncoder(initial_channel=initial_channel)
        self.encoder_12 = LeadEncoder(initial_channel=initial_channel)

        self.Classifier = ClassificationHead(input_channel=input_channel, initial_channel=32, num_classes=num_classes)

    def forward(self, x):
        ftr_1 = self.encoder_1(x[:, None, 0])
        ftr_2 = self.encoder_1(x[:, None, 1])
        ftr_3 = self.encoder_1(x[:, None, 2])
        ftr_4 = self.encoder_1(x[:, None, 3])
        ftr_5 = self.encoder_1(x[:, None, 4])
        ftr_6 = self.encoder_1(x[:, None, 5])
        ftr_7 = self.encoder_1(x[:, None, 6])
        ftr_8 = self.encoder_1(x[:, None, 7])
        ftr_9 = self.encoder_1(x[:, None, 8])
        ftr_10 = self.encoder_1(x[:, None, 9])
        ftr_11 = self.encoder_1(x[:, None, 10])
        ftr_12 = self.encoder_1(x[:, None, 11])

        x = torch.cat([ftr_1, ftr_2, ftr_3, ftr_4, ftr_5, ftr_6,
                       ftr_7, ftr_8, ftr_9, ftr_10, ftr_11, ftr_12], dim=1)

        y = self.Classifier(x)

        return y


class LeadEncoder(nn.Module):
    # use max pooling as the downsampling strategy
    def __init__(self, input_channel=1, initial_channel=32):
        super(LeadEncoder, self).__init__()

        def conv_block(inc, ouc):
            return nn.Sequential(
                nn.Conv1d(inc, ouc, 5, 1, padding=2, bias=False),
                nn.BatchNorm1d(ouc),
                nn.ReLU(inplace=True)
            )

        self.max_pooling = nn.MaxPool1d(5, stride=2)

        self.conv1 = conv_block(input_channel, initial_channel)
        self.conv2 = conv_block(initial_channel, initial_channel*2)
        self.conv3 = conv_block(initial_channel*2, initial_channel*2*2)

        # self.pointconv1 = nn.Conv1d(input_channel, initial_channel, stride=1)
        self.pointconv2 = nn.Conv1d(initial_channel, initial_channel*2, kernel_size=1)
        self.pointconv3 = nn.Conv1d(initial_channel*2, initial_channel*4, kernel_size=1)

        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pooling(x)

        residual = self.conv2(x)
        x = self.pointconv2(x)
        x += residual
        x = self.max_pooling(x)

        residual = self.conv3(x)
        x = self.pointconv3(x)
        x += residual
        x = self.max_pooling(x)
        x = torch.flatten(self.gap(x), start_dim=1)

        x = x.reshape((x.shape[0], 1, x.shape[1]))

        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_channel, initial_channel, num_classes=2, hidden_units=500):
        super(ClassificationHead, self).__init__()
        def conv_block(inc, ouc):
            return nn.Sequential(
                nn.Conv1d(inc, ouc, 5, 1, padding=2, bias=False),
                nn.BatchNorm1d(ouc),
                nn.ReLU(inplace=True)
            )

        self.max_pooling = nn.MaxPool1d(5, stride=2)

        self.conv1 = conv_block(input_channel, initial_channel)
        self.conv2 = conv_block(initial_channel, initial_channel * 2)
        self.conv3 = conv_block(initial_channel * 2, initial_channel * 2 * 2)

        # self.pointconv1 = nn.Conv1d(input_channel, initial_channel, stride=1)
        self.pointconv2 = nn.Conv1d(initial_channel, initial_channel * 2, kernel_size=1)
        self.pointconv3 = nn.Conv1d(initial_channel * 2, initial_channel * 4, kernel_size=1)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(initial_channel * 4, hidden_units)
        self.fc1 = nn.Linear(hidden_units, num_classes)
        self.drop_50 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pooling(x)

        residual = self.conv2(x)
        x = self.pointconv2(x)
        x += residual
        x = self.max_pooling(x)

        residual = self.conv3(x)
        x = self.pointconv3(x)
        x += residual
        x = self.max_pooling(x)

        features = self.gap(x)
        features = torch.flatten(features, start_dim=1)
        x = self.drop_50(features)
        y = self.fc(x)
        y = self.fc1(y)

        return y
