import torch.nn as nn
import torch
import torch.nn.functional as F


class ECGUGroupNet(nn.Module):
    def __init__(self, num_classes=2, unet_classes=2, in_channels=12, unet_initial_filter_size=32, kernel_size=5,
                 groups=12):
        super(ECGUGroupNet, self).__init__()
        self.unet = UNet1d(num_classes=unet_classes, in_channels=in_channels, initial_filter_size=unet_initial_filter_size,
                           kernel_size=kernel_size, groups=groups)

        self.classifier = Classifier(num_classes, in_channels)

    def forward(self, x):
        result = self.unet(x)
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


class UNet1d(nn.Module):
    def __init__(self, num_classes=2, in_channels=12, initial_filter_size=32, kernel_size=3, num_downs=3, groups=12,
                 norm_layer=nn.InstanceNorm1d, use_dropout=False):
        super(UNet1d, self).__init__()

        # build the innermost block
        unet_block = UNetSkpiConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs - 1),
                                             out_channels=initial_filter_size * 2 ** num_downs*groups,
                                             num_classes=num_classes, kernel_size=kernel_size, innermost=True,
                                             norm_layer=norm_layer, groups=groups)

        for i in range(1, num_downs):
            unet_block = UNetSkpiConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs - i - 1),
                                                 out_channels=initial_filter_size * 2 ** (num_downs - i)*groups,
                                                 num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block,
                                                 norm_layer=norm_layer)
        unet_block = UNetSkpiConnectionBlock(in_channels=in_channels, out_channels=initial_filter_size*groups,
                                             num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, x):
        return self.model(x)


class UNetSkpiConnectionBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, num_classes=2, kernel_size=5, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.InstanceNorm1d, use_dropout=False, groups=12):
        super(UNetSkpiConnectionBlock, self).__init__()
        self.outermost = outermost
        self.groups = groups
        # downconv
        pool = nn.MaxPool1d(2, stride=2)
        conv1 = self.contract(in_channels, out_channels, kernel_size, norm_layer)
        conv2 = self.contract(out_channels, out_channels, kernel_size, norm_layer)

        # upconv
        conv3 = self.expand(out_channels*2, out_channels, kernel_size)
        conv4 = self.expand(out_channels, out_channels, kernel_size)

        upconv = nn.ConvTranspose1d(out_channels, in_channels, kernel_size=2, stride=2, groups=groups)

        if outermost:
            final = nn.Conv1d(out_channels, num_classes*groups, kernel_size=1, groups=groups)
            down = [conv1, conv2]
            up = [conv3, conv4, final]
            model = down + [submodule] + up
        elif innermost:
            down = [pool, conv1, conv2]
            model = down + [upconv]
        else:
            down = [pool, conv1, conv2]
            up = [conv3, conv4, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=5, norm_layer=nn.InstanceNorm1d, groups=12):
         layer = nn.Sequential(
             nn.Conv1d(in_channels, out_channels, kernel_size, padding=2, groups=groups),
             norm_layer(out_channels),
             nn.LeakyReLU(inplace=True),
         )

         return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=5, groups=12):
        layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=2, groups=12),
            nn.LeakyReLU(inplace=True),
        )

        return layer

    @staticmethod
    def center_crop(layer, target_length):
        batch_size, n_channels, layer_length = layer.size()
        xy = (layer_length - target_length) // 2
        return layer[:, :, xy:(xy + target_length)]

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            crop = self.center_crop(self.model(x), x.size()[2])
            return torch.cat((x, crop), 1)


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



