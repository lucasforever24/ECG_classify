import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
import scipy.io as sio
from old_codes.network import ClassificationNetwork
import torch.utils.data as Data
from sklearn.model_selection import KFold

EPOCH = 1000   # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 10      # 批处理尺寸(batch_size)
LR = 0.001        # 学习率



A = sio.loadmat("Features.mat")
B = sio.loadmat("Classification.mat")
C = sio.loadmat("Curve.mat")
Features = torch.from_numpy(A['Features'])
Classification = torch.from_numpy(B["Classification"]).squeeze()
Curve = torch.from_numpy(C["Curve"])


parser = argparse.ArgumentParser(description='PyTorch classification Training')
parser.add_argument('--outf', default='./model/OnlyClassification/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/OnlyClassification/network.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()


best_acc = 85
model = ClassificationNetwork().cuda()
cast = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-8)



kf = KFold(n_splits=10)
kf.get_n_splits(Features)


if __name__ == '__main__':
    print("Start Training, CurveNetwork!")  # 定义遍历数据集的次数
    with open("acc(Only).txt", "w") as f:
        with open("log(Only).txt", "w")as f2:                
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                model.train(True)
            
                correct = 0
                total = 0
                #for i in range(len(Features[:,0])):
                for k, (train_index, test_index) in enumerate(kf.split(Features),0):
                    Features_train = Features[train_index]
                    Curve_train = Curve[train_index]
                    Classification_train = Classification[train_index]
                    #print(train_index)
                    #print(Features_train.size())
                    #print(Curve_train.size())
                    torch_dataset_train = Data.TensorDataset(Features_train, Classification_train)
                    torch_curve_train = Data.TensorDataset(Curve_train, Classification_train)
                    trainset = DataLoader(dataset=torch_dataset_train,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=BATCH_SIZE)
                    traincurveset = DataLoader(dataset=torch_curve_train,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=BATCH_SIZE)
                    
                    Features_test = Features[test_index]
                    Curve_test = Curve[test_index]
                    Classification_test = Classification[test_index]
                    #print(test_index)
                    #print(Features_test.size())
                    #print(Curve_test.size())
                    torch_dataset_test = Data.TensorDataset(Features_test, Classification_test)
                    torch_curve_test = Data.TensorDataset(Curve_test, Classification_test)
                    testset = DataLoader(dataset=torch_dataset_test,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=BATCH_SIZE)
                    testcurveset = DataLoader(dataset=torch_curve_test,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=BATCH_SIZE)
                    
                    
                    
                    length = len(Features_train)
                    sum_loss = 0.0
                    correct = 0.0
                    total = 0.0
                    
                    for i, (data1 , data2) in enumerate(zip(traincurveset,trainset),0):    
                        # 准备数据
                        #print(i)
                        
                        curves, targets = data1
                        features, labels = data2
                        curves , features, labels = curves.cuda(), features.cuda(), labels.cuda()
                        #print(curves.size())
                        
                        #inputs = torch.unsqueeze(inputs,0)
                        #labels = torch.unsqueeze(labels,0)
                        #features = torch.FloatTensor(len(inputs[:,0,0,0]),25).zero_()
                        #print(len(inputs[:,0,0,0]))
                        #if i == 0 :
                            #print(inputs.size())
                            #print(labels.size())
                        
                        outputs = model(curves, features).cuda()

                        #labels = labels.squeeze()
                        loss = cast(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
        

                        sum_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        #print(predicted)
                        #print(labels)
                        total += labels.size(0)
                        correct += predicted.eq(labels.data).cpu().sum()
                    
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.03f '
                        % (epoch + 1, k, sum_loss / length, (100. * correct) / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.03f '
                        % (epoch + 1, k, sum_loss / length, (100. * correct) / total))
                    f2.write('\n')
                    f2.flush()

                    #print("Waiting Test!")
                    with torch.no_grad():
                        for i, ( data1 , data2) in enumerate(zip(testcurveset,testset),0):   
                            model.eval() 
                            # 准备数据
                            #print(i)
                            curves, targets = data1
                            features, labels = data2
                            curves , features, labels = curves.cuda(), features.cuda(), labels.cuda()
                            outputs = model(curves, features).cuda()
                            #outputs = model(inputs).cuda()
                            # 取得分最高的那个类 (outputs.data的索引号)
                            #labels = labels.squeeze()
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum()
                
                print('测试分类准确率为：%.03f' % (100 * correct / total))
                acc = 100. * correct / total
                # 将每次测试结果实时写入acc.txt文件中
                print('Saving model......')
                torch.save(model.state_dict(), '%s/model_%03d.pth' % (args.outf, epoch + 1))
                f.write("EPOCH=%03d,Accuracy= %.03f" % (epoch + 1, acc))
                f.write('\n')
                f.flush()
                # 记录最佳测试分类准确率并写入best_acc.txt文件中
                if acc > best_acc:
                    f3 = open("best_acc(18).txt", "w")
                    f3.write("EPOCH=%d,best_acc= %.03f" % (epoch + 1, acc))
                    f3.close()
                    best_acc = acc

            print("Training Finished, TotalEPOCH=%d" % EPOCH)

