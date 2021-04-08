from old_codes.unet_parts import *

class NN(nn.Module):
    def __init__(self):#input_size, num_layers, num_classes):
        super(NN, self).__init__()
        
        self.layer1 = nn.Linear(30,30)
        self.layer2 = nn.Linear(30,3)

    def forward(self, x):
        
        x = self.layer1(x)
        #x = x.view(x.size(0), -1)
        out = self.layer2(x)
        #return out
        return F.log_softmax(out , dim=1)


def Network():
    return NN()


class NN2(nn.Module):
    def __init__(self):#input_size, num_layers, num_classes):
        super(NN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,2, kernel_size = 1, stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2,4, kernel_size = (1,10), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4,8, kernel_size = (1,20), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(8,16, kernel_size = (1,30), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(16,16, kernel_size = (1,40), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16,8, kernel_size = (1,50), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(8,4, kernel_size = (1,60), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(4,2, kernel_size = (1,12), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(2,1, kernel_size = (12,1), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )


        self.layer1 = nn.Linear(50,50)
        self.layer2 = nn.Linear(50,3)
        
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x1 = self.conv6(x1)
        x1 = self.conv7(x1)
        x1 = self.conv8(x1)
        x1 = self.conv9(x1)
        x1 = x1.squeeze(1)
        x1 = x1.squeeze(1)
        #print(x1.size())
        #print(x2.size())
        x = torch.cat((x1, x2), 1)
        #print(x.size())
        #print(x.size())
        x = self.layer1(x)
        #x = x.view(x.size(0), -1)
        out = self.layer2(x)
        #return out
        return F.log_softmax(out , dim=1)

def CurveNetwork():
    return NN2()



class NN3(nn.Module):
    def __init__(self):#input_size, num_layers, num_classes):
        super(NN3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,12, kernel_size = 1, stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(12,24, kernel_size = (1,10), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(24,48, kernel_size = (1,20), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(48,96, kernel_size = (1,30), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(96,96, kernel_size = (1,40), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(96,48, kernel_size = (1,50), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(48,24, kernel_size = (1,60), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(24,12, kernel_size = (1,7), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(12,1, kernel_size = (12,1), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )


        self.layer1 = nn.Linear(60,60)
        self.layer2 = nn.Linear(60,30)
        self.layer3 = nn.Linear(30,15)     
        self.layer4 = nn.Linear(15,3)
        
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x1 = self.conv6(x1)
        x1 = self.conv7(x1)
        x1 = self.conv8(x1)
        x1 = self.conv9(x1)
        x1 = x1.squeeze(1)
        x1 = x1.squeeze(1)
        #print(x1.size())
        #print(x2.size())
        x = torch.cat((x1,x2),1)
        #print(x.size())
        #print(x.size())
        x = self.layer1(x)
        #x = x.view(x.size(0), -1)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.layer4(x)
        #return out
        return F.log_softmax(out , dim=1)

def ClassificationNetwork():
    return NN3()



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = inconv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 1)
        self.layer1 = nn.Linear(67,60)
        self.layer2 = nn.Linear(60,30)
        self.layer3 = nn.Linear(30,15)     
        self.layer4 = nn.Linear(15,3)


        self.conv1 = nn.Sequential(
            nn.Conv2d(1,12, kernel_size = 1, stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(12,24, kernel_size = (1,10), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(24,48, kernel_size = (1,20), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(48,96, kernel_size = (1,30), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(96,96, kernel_size = (1,40), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(96,48, kernel_size = (1,50), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(48,24, kernel_size = (1,60), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(24,12, kernel_size = (1,7), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(12,1, kernel_size = (12,1), stride = 1, padding = 0,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x, y):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.squeeze(1)
        x = x.squeeze(1)
        #print(x.size())
        #print(y.size())
        x = torch.cat((x,y),1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.layer4(x)
        return F.log_softmax(out , dim=1)




