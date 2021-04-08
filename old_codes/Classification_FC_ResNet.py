import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
import scipy.io as sio
from old_codes.ResNet import resnet18
import torch.utils.data as Data
from sklearn.model_selection import KFold
import csv



EPOCH = 1000   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 40      #批处理尺寸(batch_size)
LR = 0.001        #学习率



A = sio.loadmat("Train_Features.mat")
B = sio.loadmat("Train_Classification.mat")
C = sio.loadmat("Train_Curve.mat")

D = sio.loadmat("Test_Features.mat")
E = sio.loadmat("Test_Classification.mat")
F = sio.loadmat("Test_Curve.mat")



Features = torch.from_numpy(A['Features'])
Classification = torch.from_numpy(B["Classification"]).squeeze()
Curve = torch.from_numpy(C["Curve"])

Features_t = torch.from_numpy(D['Features'])
Classification_t = torch.from_numpy(E["Classification"]).squeeze()
Curve_t = torch.from_numpy(F["Curve"])



parser = argparse.ArgumentParser(description='PyTorch classification Training')
parser.add_argument('--outf', default='./model/Classification_FC_ResNet/', help='folder to output images and model checkpoints') #输出结果保存路径
#parser.add_argument('--net', default='./model/Classification_FC_Unet/network.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()


best_acc = 85
model = resnet18().cuda()
cast = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-8)


n = 10
kf = KFold(n_splits=n)
kf.get_n_splits(Features)


if __name__ == '__main__':
    print("Start Training, ResNet Network!")  # 定义遍历数据集的次数   
    
    with open("Result(Classification_FC_ResNet). csv", "a")as f:  
        writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(["epoch","loss","average","validation","final"])


    final_dataset_test = Data.TensorDataset(Features_t, Classification_t)
    final_curve_test = Data.TensorDataset(Curve_t, Classification_t)
    finaldataset = DataLoader(dataset=final_dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=BATCH_SIZE)
    finalcurveset = DataLoader(dataset=final_curve_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=BATCH_SIZE)


    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train(True)
        sum_loss = 0.0
        correct = torch.tensor(0).double()
        total = torch.tensor(0).double()
        correct_t = torch.tensor(0).double()
        total_t = torch.tensor(0).double()
        loss_final = torch.tensor(0).double()
        average = torch.tensor(0).double()
        validation = torch.tensor(0).double()
        #for i in range(len(Features[:,0])):
        for train_index, test_index in kf.split(Features):
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
                    total_t += labels.size(0)
                    correct_t += (predicted == labels).cpu().sum()
            
            
        loss_final = sum_loss / (n * length)
        average = (100. * correct) / total
        validation = (100. * correct_t) / total_t
        print('[epoch:%d] Loss: %.03f | Average: %.3f%% '% (epoch + 1, loss_final , average))
        
        print('验证分类准确率为：%.3f%%' % validation)
        #print("validation_epoch: "+str(epoch+1))
        #print("validation: "+str(validation))


        final_total = torch.tensor(0).double()
        final_correct = torch.tensor(0).double()
        final = torch.tensor(0).double()
        with torch.no_grad():
            for i, ( data1 , data2) in enumerate(zip(finalcurveset,finaldataset),0):   
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
                final_total += labels.size(0)
                final_correct += (predicted == labels).cpu().sum()

            final = 100. * final_correct / final_total
        print('测试集分类准确率为：%.3f%%' % final)
            

        with open("Result(Classification_FC_RstNet).csv", "a")as f:  
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow([epoch,loss_final,average,validation,final])
            # 将每次测试结果实时写入acc.txt文件中
            print('Saving model......')
            torch.save(model.state_dict(), '%s/model_%03d.pth' % (args.outf, epoch + 1))

            
            # 记录最佳测试分类准确率并写入best_acc.txt文件中
        
        if final > best_acc:
            f2 = open("best_acc(Classification_FC_ResNet).csv", "a")
            writer = csv.writer(f2, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow([epoch,final])
            f2.close()
            best_acc = final
        
    print("Training Finished, TotalEPOCH=%d" % EPOCH)

