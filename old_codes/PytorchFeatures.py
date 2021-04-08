import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import argparse
from old_codes.network import Network
import torch.utils.data as Data

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ExtractPixel(im,m,n):
    ECG = im  #.cuda()
    line = 0
    line_extreme_up = 0
    #line_extreme_down = 0
    before_count_up = 0
    no_area = 0
    #before_count_down = 0
    line1_extreme = torch.Tensor([])
    line3_extreme = torch.Tensor([])
    line_candidate1 = torch.Tensor([])
    line_candidate2 = torch.Tensor([])
    line_candidate3 = torch.Tensor([])
    line1 = torch.IntTensor(240).zero_()
    line2 = torch.IntTensor(240).zero_()
    line3 = torch.IntTensor(240).zero_()
    #print(line3)
    tem_counts1 = 0
    tem_counts2 = 0
    tem_counts3 = 0

    for j in range(m, n):
        for i in range(181,670):
            if ECG[i][j] == 0 and line == 0:
                #print(ECG[i][j])
                #print(str(j)+ " " + str(i))
                line_candidate1 = torch.cat((line_candidate1,torch.Tensor([669-i])))
                #print(line_candidate1.size())
                if i == 181:
                    if before_count_up == 0:
                        line_extreme_up = 1
                        line1_extreme = torch.cat((line1_extreme,torch.Tensor([j])))
                    else :
                        line_extreme_up = 1
                        before_count_up = 1    

            elif (i >= 182 ) and (ECG[i][j] != 0) and (ECG[i-1][j] == 0) and (line == 0) :
                line = line + 1
                continue
            if ECG[i][j] == 0 and line == 1:
                line_candidate2 = torch.cat((line_candidate2,torch.Tensor([669-i])))
                '''
                if len(line_candidate2) < 1:
                    line_candidate2.append(635 - i) 
                elif (line_candidate2[len(line_candidate2)-1] - (635 - i)) == 1:
                    line_candidate2.append(635 - i) 
                '''
            elif (ECG[i][j] != 0) and (ECG[i-1][j] == 0) and (line == 1) :
                line = line + 1
                continue
            if ECG[i][j] == 0 and line == 2:
                line_candidate3 = torch.cat((line_candidate3,torch.Tensor([669-i]))) 
                '''
                if len(line_candidate3) < 1:
                    line_candidate3.append(669 - i) 
                elif (line_candidate3[len(line_candidate3)-1] - (669 - i)) == 1:
                    line_candidate3.append(669 - i)
                '''
            elif (ECG[i][j] != 0) and (ECG[i-1][j] == 0) and (line == 2):
                line = 0
                break

            
        x = j - m
        #print(x)
        #print(line_candidate1)
        #print(line_candidate2)
        #print(line_candidate3)
        #返回众数

        #tem_counts1 = torch.mode(line1)  
        #tem_counts2 = torch.mode(line2)  
        #tem_counts3 = torch.mode(line3)  
    

        #print(len(tem_counts3))
        #print(line_candidate2.size())
        #print(len(line_candidate2))
        if len(line_candidate2) == 0:
            line_candidate2 = line_candidate1
            line_candidate3 = line_candidate1

        elif len(line_candidate3) == 0:
            line_candidate3 = line_candidate2

        
        if ((x == 0) or (x == 1)) :

            if (line_extreme_up == 1 and before_count_up == 0): #and (begin == 0) :
                line1[x] = torch.max(line_candidate1)
                line2[x] = torch.min(line_candidate2)
                line3[x] = torch.min(line_candidate3)
                before_count_up = 1
                #before_count_down = 0
                line_extreme_up = 0
                #line_extreme_down = 0
                #begin = 1
                
                
            elif (line_extreme_up == 0 and before_count_up == 1): #and (begin == 1):
                line1[x] = torch.Tensor([669 - 181])
                line2[x] = torch.min(line_candidate1)
                line3[x] = torch.min(line_candidate2)
                before_count_up = 1
                before_count_down = 0
                line_extreme_up = 0
                line_extreme_down = 0


            elif (line_extreme_up == 1 and before_count_up == 1): #and (begin == 1) :
                line1[x] = torch.max(line_candidate1)
                line2[x] = torch.min(line_candidate2)
                line3[x] = torch.min(line_candidate3)
                before_count_up = 0
                #before_count_down = 0
                line_extreme_up = 0
                #line_extreme_down = 0
                #begin = 0


            elif (line_extreme_up == 0 and before_count_up == 0):
                line1[x] = torch.min(line_candidate1)
                line2[x] = torch.min(line_candidate2)
                line3[x] = torch.min(line_candidate3)
        
        else:
            if (line_extreme_up == 1 and before_count_up == 0):
                line1[x] = torch.Tensor([669 - 181])
                if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                    line2[x] = torch.max(line_candidate2)
                elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                    line2[x] = torch.min(line_candidate2)
                elif (line2[x-2]-line2[x-1] == 0):
                    if (line2[x-3]-line2[x-2] < 0):
                        line2[x] = torch.max(line_candidate2)
                    elif (line2[x-3]-line2[x-2] > 0):
                        line2[x] = torch.min(line_candidate2)
                    else :
                        line2[x] = torch.min(line_candidate2)                
                
                if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                    line3[x] = torch.max(line_candidate3)
                elif (line3[x-2]-line3[x-1] > 0):   
                    line3[x] = torch.min(line_candidate3)
                elif (line3[x-2]-line3[x-1] == 0):
                    if (line3[x-3]-line3[x-2] < 0):
                        line3[x] = torch.max(line_candidate3)
                    elif (line3[x-3]-line3[x-2] > 0):
                        line3[x] = torch.min(line_candidate3)
                    else :
                        line3[x] = torch.min(line_candidate3)


                if (line3[x-1] >= line3[x] + 120):
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = torch.max(line_candidate2)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = torch.min(line_candidate2)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = torch.min(line_candidate2)
                        else :
                            line3[x] = torch.min(line_candidate2)
                    
                    #if line1[x] == (669 - 181 - 25) or 
                if (line2[x-1] >= line2[x] + 120):
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = torch.max(line_candidate1)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = torch.min(line_candidate1)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = torch.min(line_candidate1)
                        else :
                            line2[x] = torch.min(line_candidate1)


                if (line2[x] == line1[x]):
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = torch.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = torch.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = torch.min(line_candidate2)
                        else :
                            line2[x] = torch.min(line_candidate2)


                if (line3[x] == line2[x]):
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = torch.max(line_candidate3)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = torch.min(line_candidate3)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = torch.min(line_candidate3)
                        else :
                            line3[x] = torch.min(line_candidate3)

                before_count_up = 1
                #before_count_down = 1
                line_extreme_up = 0
                #line_extreme_down = 0
            
            elif (line_extreme_up == 0 and before_count_up == 1):
                line1[x] = torch.Tensor([669 - 181])
                if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                    line2[x] = torch.max(line_candidate1)
                elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                    line2[x] = torch.min(line_candidate1)
                elif (line2[x-2]-line2[x-1] == 0):
                    if (line2[x-3]-line2[x-2] < 0):
                        line2[x] = torch.max(line_candidate1)
                    elif (line2[x-3]-line2[x-2] > 0):
                        line2[x] = torch.min(line_candidate1)
                    else :
                        line2[x] = torch.min(line_candidate1)

                if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                    line3[x] = torch.max(line_candidate2)
                elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                    line3[x] = torch.min(line_candidate2)
                elif (line3[x-2]-line3[x-1] == 0):
                    if (line3[x-3]-line3[x-2] < 0):
                        line3[x] = torch.max(line_candidate2)
                    elif (line3[x-3]-line3[x-2] > 0):
                        line3[x] = torch.min(line_candidate2)
                    else :
                        line3[x] = torch.min(line_candidate2)


                if (line3[x-1] >= line3[x] + 120):
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = torch.max(line_candidate1)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = torch.min(line_candidate1)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = torch.min(line_candidate1)
                        else :
                            line3[x] = torch.min(line_candidate1)

                '''  
                if (line2[x] == line1[x]):
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = torch.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = torch.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = torch.min(line_candidate2)
                        else :
                            line2[x] = torch.min(line_candidate2)
                '''

                if (line3[x] == line2[x]):
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = torch.max(line_candidate2)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = torch.min(line_candidate2)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = torch.min(line_candidate2)
                        else :
                            line3[x] = torch.min(line_candidate2)


                before_count_up = 1
                #before_count_down = 0
                line_extreme_up = 0
                #line_extreme_down = 0
            
            elif (line_extreme_up == 1 and before_count_up == 1):
                line1[x] = torch.Tensor([669 - 181])


                if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                    line2[x] = torch.max(line_candidate2)
                elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                    line2[x] = torch.min(line_candidate2)
                elif (line2[x-2]-line2[x-1] == 0):
                    if (line2[x-3]-line2[x-2] < 0):
                        line2[x] = torch.max(line_candidate2)
                    elif (line2[x-3]-line2[x-2] > 0):
                        line2[x] = torch.min(line_candidate2)
                    else :
                        line2[x] = torch.min(line_candidate2)


                if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                    line3[x] = torch.max(line_candidate3)
                elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                    line3[x] = torch.min(line_candidate3)
                elif (line3[x-2]-line3[x-1] == 0):
                    if (line3[x-3]-line3[x-2] < 0):
                        line3[x] = torch.max(line_candidate3)
                    elif (line3[x-3]-line3[x-2] > 0):
                        line3[x] = torch.min(line_candidate3)
                    else :
                        line3[x] = torch.min(line_candidate3)
                
                
                if (line3[x-1] >= line3[x] + 120):
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = torch.max(line_candidate2)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = torch.min(line_candidate2)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = torch.min(line_candidate2)
                        else :
                            line3[x] = torch.min(line_candidate2)

                    
                    #if line1[x] == (669 - 181 - 25) or 
                if (line2[x-1] >= line2[x] + 120):
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = torch.max(line_candidate1)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = torch.min(line_candidate1)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = torch.min(line_candidate1)
                        else :
                            line2[x] = torch.min(line_candidate1)


                if (line2[x] == line1[x]):
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = torch.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = torch.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = torch.min(line_candidate2)
                        else :
                            line2[x] = torch.min(line_candidate2)


                if (line3[x] == line2[x]):
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = torch.max(line_candidate3)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = torch.min(line_candidate3)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = torch.min(line_candidate3)
                        else :
                            line3[x] = torch.min(line_candidate3)




                line_extreme_up = 0
                #line_extreme_down = 0
                before_count_up = 0
                #before_count_down = 1


            elif (line_extreme_up == 0 and before_count_up == 0):
                if (line1[x-1] - line2[x-1] >=0 ) and (line1[x-1] - line2[x-1] < (len(line_candidate1) + 7)) and ((torch.min(line_candidate1) - torch.max(line_candidate2)) >= 50) : 
                    line1[x] = torch.min(line_candidate1)
                    line2[x] = torch.max(line_candidate1)
                    #print(line1[x-1] - line2[x-1])
                    #print(len(line_candidate1))
                    
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = torch.max(line_candidate2)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = torch.min(line_candidate2)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = torch.max(line_candidate2)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = torch.min(line_candidate2)
                        else :
                            line3[x] = torch.min(line_candidate2)   
                        
                    if (line3[x-1] >= line3[x] + 120 ):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = torch.max(line_candidate1)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = torch.min(line_candidate1)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = torch.min(line_candidate1)
                            else :
                                line3[x] = torch.min(line_candidate1)
                

                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = torch.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = torch.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = torch.min(line_candidate2)
                            else :
                                line3[x] = torch.min(line_candidate2)


                elif (line2[x-1] - line3[x-1] >=0 ) and (line2[x-1] - line3[x-1] < (len(line_candidate2) + 7)) and ((torch.min(line_candidate2) - torch.max(line_candidate3)) >= 50) : 
                    #print(line2[x-1] - line3[x-1])
                    #print(len(line_candidate1))
                    if (line1[x-2]-line1[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line1[x] = torch.max(line_candidate1)    
                    elif (line1[x-2]-line1[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line1[x] = torch.min(line_candidate1)
                    elif (line1[x-2]-line1[x-1] == 0):
                        if (line1[x-3]-line1[x-2] < 0):
                            line1[x] = torch.max(line_candidate1)
                        elif (line1[x-3]-line1[x-2] > 0):
                            line1[x] = torch.min(line_candidate1)
                        else :
                            line1[x] = torch.min(line_candidate1)
                    
                    line2[x] = torch.min(line_candidate2)
                    line3[x] = torch.max(line_candidate2)              
                
                else:
                    if (line1[x-2]-line1[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line1[x] = torch.max(line_candidate1)
                        
                    elif (line1[x-2]-line1[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line1[x] = torch.min(line_candidate1)
                    elif (line1[x-2]-line1[x-1] == 0):
                        if (line1[x-3]-line1[x-2] < 0):
                            line1[x] = torch.max(line_candidate1)
                        elif (line1[x-3]-line1[x-2] > 0):
                            line1[x] = torch.min(line_candidate1)
                        else :
                            line1[x] = torch.min(line_candidate1)


                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = torch.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = torch.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = torch.max(line_candidate2)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = torch.min(line_candidate2)
                        else :
                            line2[x] = torch.min(line_candidate2)

                    
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = torch.max(line_candidate3)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = torch.min(line_candidate3)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = torch.max(line_candidate3)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = torch.min(line_candidate3)
                        else :
                            line3[x] = torch.min(line_candidate3)


                    #if (line2[x] == torch.max(line_candidate1)) or  (line2[x] == torch.min(line_candidate1)) or 
                    if (line3[x-1] >= line3[x] + 120 ):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = torch.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = torch.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = torch.min(line_candidate2)
                            else :
                                line3[x] = torch.min(line_candidate2)

                    
                    #if (line1[x] == 669 - 181) or 
                    if (line2[x-1] >= line2[x] + 120 ):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = torch.max(line_candidate1)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = torch.min(line_candidate1)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = torch.min(line_candidate1)
                            else :
                                line2[x] = torch.min(line_candidate1)

                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = torch.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = torch.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = torch.min(line_candidate2)
                            else :
                                line2[x] = torch.min(line_candidate2)


                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = torch.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = torch.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = torch.min(line_candidate3)
                            else :
                                line3[x] = torch.min(line_candidate3)

                before_count_up = 0
                before_count_down = 0
                line_extreme_up = 0
                line_extreme_down = 0
        
        line = 0
        tem_counts1 = 0
        tem_counts2 = 0
        tem_counts3 = 0
        line_candidate1 = torch.Tensor([])
        line_candidate2 = torch.Tensor([])
        line_candidate3 = torch.Tensor([])

    #print(line1)

    return line1, line2, line3 


def ExtractFeatures(high,name):
    T1 = torch.zeros(7)
    T2 = torch.zeros(7)
    V1 = 0
    V2 = 0
    V3 = 0
    V4 = 0
    V5 = 0
    V6 = 0
    Positive1 = 0
    Positive2 = 0
    uptype = 0
    downtype = 0 
    curve = 0
    V1_max = 0
    T1_sum = 0
    T2_sum = 0
    T1_final = float(0)
    T2_final = float(0)
    I_minus = 0
    II_minus = 0
    III_minus = 0
    area = 0
    area_sum = 0
    area_final = float(0)
    square = 0
    aVL_max = 0
    PvN = float(1)
    #High = torch.zeros(120)
    #print('Picture ' + name[k] + ':')

    for k in range(12):
        Q_final = 0
        S_final = 0
        #A = 0
        #A1 = 0
        #A2 = 0
        B = 0
        C = 0

        A = torch.Tensor([])
        A1 = torch.Tensor([])
        A2= torch.Tensor([])
        #print('Picture ' + name[k] + ':')
        #sio.savemat("line"+str(k)+".mat",{"line"+str(k):High})
        High = high[k]#.cuda()
        '''
        for i in range (240):
            print(i)
            print(High[i])
        '''
        for i in range(2,237):
            if (abs(High[i+1]-High[i]) + abs(High[i-1]-High[i])) >= 100:
                High[i] = (High[i+1] + High[i-1])/2
                #print("good")
        #sio.savemat("High"+str(k)+".mat",{"High":High})
        A1 = int(torch.max(High)) 
        #print(A1)
        #A1 = torch.max(High)
        A2 = int(torch.min(High))
        #print(A2)
        #A2 = torch.min(High)
            #返回众数
        #A = torch.mode(High)
        A = int(torch.mode(High)[0])
        
        #print("Up or down: " + str(B)+"," + str(C))

        #G =  - A2
        #M = A1 - base
        #R_index1 = 0
        #R_index2 = 0
        #print(High)
        down = 0
        #print(base)
        if A - A2 > A1 - A:
            #print('down')
            downtype = 1
            if name[k] == 'II':
                II = 0
            if name[k] == 'III':
                III = 0
            if name[k] == 'V1':
                V1 = 0
            if name[k] == 'V2':
                V2 = 0
            if name[k] == 'V3':
                V3 = 0 
            if name[k] == 'V4':
                V4 = 0 
            if name[k] == 'V5':
                V5 = 0 
            if name[k] == 'V6':
                V6 = 0 

            if k > 6 : 
                continue#break
                
                
            #print(torch.where(High==1))
            #print(High_extreme_down)
            #print(C)
                    
            

            
            
            R_index = int(torch.min(High,0)[1])
            
            while R_index <= 50 or R_index >=200:
                #print("A"+ str(A))
                #print("R_index"+str(R_index))
                #print(int(torch.min(High,0)[1]))
                #print(A)
                High[R_index] = torch.Tensor([A])
                #print(High[R_index]) 
                #print(High[R_index])
                New_index = int(torch.min(High,0)[1])
                #print("New_index"+ str(New_index))
                R_index = New_index
                #print(R_index)
            R_index1 = R_index - 2
            R_index2 = R_index + 2
            if int(torch.max(High))>=668:
                #R_index1 = R_index - 2
                R_index2 = R_index + 6   ########!!!!!!change the number
            if R_index1 - 20 >= 0 :
                downrange = R_index1 - 20#######!!!!!change the 15 to 20
            else:
                downrange = 0
            if R_index2 + 20 <= 239 :
                uprange = R_index2 + 20
            else:
                uprange = 239

            base = 0
            base = int(torch.mode(High[R_index1-40:R_index2+20])[0])###!!!!!change
            #print(Base)
            
            if base == 0:
                #Base = torch.bincount(High)###!!!!!change
                base = int(torch.mode(High)[0])
            #print(base)
            #print(R_index1)
            #print(R_index2)
            
            
            #the real high
            High = high[k]#.cuda()
            for i in range(2,237):
                if (abs(High[i+1]-High[i]) + abs(High[i-1]-High[i])) >= 100:
                    High[i] = (High[i+1] + High[i-1])/2


            '''''''''''''''
            down
            Find S extreme
            '''''''''''''''

            S_index = 0
            for i in range(R_index2, uprange):
                if (i + 2 <= 239) and (i - 2 >= 0 ):
                    if (High[i-2]<High[i]) and (High[i+2]<High[i]):
                        S_index = i
                        break
                    elif ((High[i-2]<High[i]) and (High[i+2]<=High[i])) or ((High[i-2]<=High[i]) and (High[i+2]<High[i])):
                        S_index = i
                        break
                    elif (High[i-2]<=High[i]) and (High[i+2]<=High[i]):
                        S_index = i
                        break
                    elif ((High[i-2]<High[i]) and (High[i+1]<High[i])) or ((High[i-1]<High[i]) and (High[i+2]<High[i])):
                        S_index = i
                        break
                    elif (((High[i-2]<=High[i]) and (High[i+1]<High[i])) or ((High[i-2]<High[i]) and (High[i+1]<=High[i]))) or (((High[i-1]<=High[i]) and (High[i+2]<High[i])) or ((High[i-1]<High[i]) and (High[i+2]<=High[i]))):
                        S_index = i
                        break
                    elif (((High[i-2]<=High[i]) and (High[i+1]<=High[i])) or ((High[i-1]<=High[i]) and (High[i+2]<=High[i]))):
                        S_index = i
                        break
                    elif ((High[i-1]<High[i]) and (High[i+1]<High[i])):
                        S_index = i
                        break
                    elif ((High[i-1]<=High[i]) and (High[i+1]<High[i])) or ((High[i-1]<High[i]) and (High[i+1]<=High[i])):
                        S_index = i
                        break
                    elif ((High[i-1]<=High[i]) and (High[i+1]<=High[i])):
                        S_index = i
                        break
            
                elif ((i + 1 <= 239) and (i - 1 >= 0 )):
                    if ((High[i-1]<High[i]) and (High[i+1]<High[i])):
                        S_index = i
                        break
                    elif ((High[i-1]<=High[i]) and (High[i+1]<High[i])) or ((High[i-1]<High[i]) and (High[i+1]<=High[i])):
                        S_index = i
                        break
                    elif ((High[i-1]<=High[i]) and (High[i+1]<=High[i])):
                        S_index = i
                        break
            
            if S_index == 0:
                S_index = R_index2 + 3
            #if len(S_index)>3:
            #    S = int(torch.min(S_index))
            #else:
            #    S = int(torch.median(S_index))

            '''''''''''''''
            down
            Find S_final
            '''''''''''''''
            ########add many different cases
            S_final = 0
            for i in range(S_index, S_index + 15 ):
                #print("i = "+ str(i))
                F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2]) 
                if (F <= 1) and ( abs(base - High[i]) <= 10 ):
                    S_final = i
                    break
            
            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2]) 
                    #print(F)
                    if (F <= 2) and ( abs(base - High[i]) <=10 ):
                        S_final = i
                        break

            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2]) 
                    #print(F)
                    if (F <= 1) and ( abs(base - High[i]) <= 20 ):
                        S_final = i
                        break
                    #print(S_final)
                    #print('Hello again')
                    
            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2]) 
                    #print(F)
                    if (F <= 2) and ( abs(base - High[i]) <= 20 ):
                        S_final = i
                        break
                    #print(S_final)
                    #print('Hello again')   

            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2]) 
                    #print(F)
                    
                    if (F <= 3) and ( abs(base - High[i]) <= 20 ):
                        S_final = i
                        break
                    #print(S_final)
                    #print('Hello again')
                    
        
            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2]) 
                    #print(F)
                    if (F <= 2) and ( abs(base - High[i]) <= 50 ):
                        S_final = i
                        break
                    #print(S_final)
                    #print('Hello again')
                    

            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    if F >= base:
                        S_final = i
                        break
                    #print(S_final)
                    #print('Hello again')
                    

            if S_final == 0:
                S_final = S_index
                for i in range(S_index, S_index + 15):
                    if High[i] > High[S_final]:
                        S_final = i
                #break     #####!!!!!!!Add new break
            
            if S_final == 0:
                S_final = S_index + 3
            '''''''''''''''
            down
            Find Q extreme
            '''''''''''''''

            
            Q_index = 0
            for i in range(R_index1, downrange, -1):
                #print(abs(High[i] - High[S_index]))
                #print(str(i) + ", " + str(High[i]))
                if (i + 2 <= 239) and (i - 2 >= 0 ):
                    if (abs(High[i] - base) < 20):   ##########!!!!!!change the S_index to base
                        
                        if (High[i-2]<High[i]) and (High[i+2]<High[i]):
                            Q_index = i
                            break
                        elif ((High[i-2]<High[i]) and (High[i+2]<=High[i])) or ((High[i-2]<=High[i]) and (High[i+2]<High[i])):
                            Q_index = i
                            break
                        elif (High[i-2]<=High[i]) and (High[i+2]<=High[i]):
                            Q_index = i
                            break
                        elif ((High[i-2]<High[i]) and (High[i+1]<High[i])) or ((High[i-1]<High[i]) and (High[i+2]<High[i])):
                            Q_index = i
                            break
                        elif (((High[i-2]<=High[i]) and (High[i+1]<High[i])) or ((High[i-2]<High[i]) and (High[i+1]<=High[i]))) or (((High[i-1]<=High[i]) and (High[i+2]<High[i])) or ((High[i-1]<High[i]) and (High[i+2]<=High[i]))):
                            Q_index = i
                            break
                        elif (((High[i-2]<=High[i]) and (High[i+1]<=High[i])) or ((High[i-1]<=High[i]) and (High[i+2]<=High[i]))):
                            Q_index = i
                            break
                        
                        if ((High[i-1]<High[i]) and (High[i+1]<High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]<=High[i]) and (High[i+1]<High[i])) or ((High[i-1]<High[i]) and (High[i+1]<=High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]<=High[i]) and (High[i+1]<=High[i])):
                            Q_index = i
                            break
                elif (i + 1 <= 239) and (i - 1 >= 0 ):
                    if (abs(High[i] - base) < 20):
                        if ((High[i-1]<High[i]) and (High[i+1]<High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]<=High[i]) and (High[i+1]<High[i])) or ((High[i-1]<High[i]) and (High[i+1]<=High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]<=High[i]) and (High[i+1]<=High[i])):
                            Q_index = i
                            break


            ##########!add one more case
            if Q_index == 0:
                if (i + 2 <= 239) and (i - 2 >= 0 ):
                    if (abs(High[i] - base) < 30):
                        
                        if (High[i-2]<High[i]) and (High[i+2]<High[i]):
                            Q_index = i
                            break
                        elif ((High[i-2]<High[i]) and (High[i+2]<=High[i])) or ((High[i-2]<=High[i]) and (High[i+2]<High[i])):
                            Q_index = i
                            break
                        elif (High[i-2]<=High[i]) and (High[i+2]<=High[i]):
                            Q_index = i
                            break
                        elif ((High[i-2]<High[i]) and (High[i+1]<High[i])) or ((High[i-1]<High[i]) and (High[i+2]<High[i])):
                            Q_index = i
                            break
                        elif (((High[i-2]<=High[i]) and (High[i+1]<High[i])) or ((High[i-2]<High[i]) and (High[i+1]<=High[i]))) or (((High[i-1]<=High[i]) and (High[i+2]<High[i])) or ((High[i-1]<High[i]) and (High[i+2]<=High[i]))):
                            Q_index = i
                            break
                        elif (((High[i-2]<=High[i]) and (High[i+1]<=High[i])) or ((High[i-1]<=High[i]) and (High[i+2]<=High[i]))):
                            Q_index = i
                            break
                        
                        if ((High[i-1]<High[i]) and (High[i+1]<High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]<=High[i]) and (High[i+1]<High[i])) or ((High[i-1]<High[i]) and (High[i+1]<=High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]<=High[i]) and (High[i+1]<=High[i])):
                            Q_index = i
                            break
                elif (i + 1 <= 239) and (i - 1 >= 0 ):
                    if (abs(High[i] - base) < 30):
                        if ((High[i-1]<High[i]) and (High[i+1]<High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]<=High[i]) and (High[i+1]<High[i])) or ((High[i-1]<High[i]) and (High[i+1]<=High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]<=High[i]) and (High[i+1]<=High[i])):
                            Q_index = i
                            break


            if Q_index == 0:
                Q_index = R_index1 - 3
            #print("Q_index: " + str(Q_index))
            #if len(Q_index)>3:
            #    Q = int(torch.max(Q_index))
            #else:
            #    Q = int(torch.median(Q_index))

            '''''''''''''''
            down
            Find Q_final
            '''''''''''''''
            Q_final = 0
            D = 0
            for i in range(Q_index, Q_index - 15 ,-1):
                D = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3])
                if High[i] <= base or D <= 1:
                    Q_final = i
                    break
            
            if Q_final == 0:
                Q_final = Q_index - 3
            
                '''
                D = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3])
                if (D <= 1) and ( abs(High[S_final] - High[i]) <= 20 ):
                    Q_final = i
                    #print('Hello')
                    break
                elif (D <= 2) and ( abs(High[S_final] - High[i]) <= 20 ):
                    Q_final = i
                    #print('Hello')
                    break
                    #print("Q_final: "+ str(Q_final))
                '''

            '''''''''''''''
            down
            Find Qs
            '''''''''''''''
            
            #find QS extreme
            R = 0
            num1 = 0
            num2 = 0
            up = 0
            Qs = 0
            for i in range(Q_final  , Q_final - 20 ,-1):
                R = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3])
                #print (R)
                #print (num)
                if name[k] == 'aVR':
                    if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) :#and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0):
                        num1 = num1 + 1
                    if num1 >=3:
                        up = 1
                    if High[i-1]-High[i]>0 and up == 1:
                    #if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0) and up==1:
                        num2 = num2 + 1
                    if num2 >=3:
                        Qs = i  ######changed from Qs = i + 3
                        break
                else :

                    #################小波应该为负############
                    if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) :#and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0):
                        num1 = num1 + 1
                    if num1 >=3:
                        up = 1
                    #print(up)
                    #if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0) and up==1:
                    if High[i-1]-High[i]>0 and up == 1:
                        num2 = num2 + 1
                    if num2 >=3:
                        Qs = i  ######changed from Qs = i + 3
                        break

            if Qs == 0:
                for i in range(Q_final  , Q_final - 20 ,-1):
                    R = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3])
                    #print (R)
                    #print (num)
                    if name[k] == 'aVR':
                        if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) :#and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0):
                            num1 = num1 + 1
                        if num1 >=3:
                            up = 1
                        if High[i-1]-High[i]>0 and up == 1:
                        #if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0) and up==1:
                            num2 = num2 + 1
                        if num2 >=2:
                            Qs = i  ######changed from Qs = i + 3
                            break
                    else :
                        if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) :#and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0):
                            num1 = num1 + 1
                        if num1 >=3:
                            up = 1
                        #print(up)
                        #if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0) and up==1:
                        if High[i-1]-High[i]>0 and up == 1:
                            num2 = num2 + 1
                        if num2 >=2:
                            Qs = i  ######changed from Qs = i + 3
                            break
            if Qs == 0:
                for i in range(Q_final  , Q_final - 20 ,-1):
                    R = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3])
                    #print (R)
                    #print (num)
                    if name[k] == 'aVR':
                        if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) :#and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0):
                            num1 = num1 + 1
                        if num1 >=2:
                            up = 1
                        if High[i-1]-High[i]>0 and up == 1:
                        #if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0) and up==1:
                            num2 = num2 + 1
                        if num2 >=3:
                            Qs = i  ######changed from Qs = i + 3
                            break
                    else :
                        if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) :#and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0):
                            num1 = num1 + 1
                        if num1 >=2:
                            up = 1
                        #print(up)
                        #if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0) and up==1:
                        if High[i-1]-High[i]>0 and up == 1:
                            num2 = num2 + 1
                        if num2 >=3:
                            Qs = i  ######changed from Qs = i + 3
                            break
                    
            if Qs == 0:
                for i in range(Q_final  , Q_final - 20 ,-1):
                    R = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3])
                    #print (R)
                    #print (num)
                    if name[k] == 'aVR':
                        if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) :#and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0):
                            num1 = num1 + 1
                        if num1 >=2:
                            up = 1
                        if High[i-1]-High[i]>0 and up == 1:
                        #if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0) and up==1:
                            num2 = num2 + 1
                        if num2 >=2:
                            Qs = i  ######changed from Qs = i + 3
                            break

                    else :
                        if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) :#and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0):
                            num1 = num1 + 1
                        if num1 >=2:
                            up = 1
                        #print(up)
                        #if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0) and up==1:
                        if High[i-1]-High[i]>0 and up == 1:
                            num2 = num2 + 1
                        if num2 >=2:
                            Qs = i  ######changed from Qs = i + 3
                            break

            if Qs == 0:
                Qs = Q_final - 3
            
            num1 = 0
            num2 = 0
            up = 0
                  

            #print("Qs: "+ str(Qs))
            #Qs = int(torch.median(Qs))



            '''''''''''''''
            down
            Find Qt
            '''''''''''''''   
            K1 = 0 
            K2 = 0
            K3 = 0
            M = 0
            Qt = 0

            
            for i in range(Qs, Qs - 10 ,-1):
                H = 0
                K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])
                #print("K1:"+str(K1))
                #print("K2:"+str(K2))
                #print("K3:"+str(K3))
                    #if K >=5 :

                        #M = M + 1

                    #if M >= 5 :    
                if K1 <= 2  and K2 <= 1 :#and K3 <= 1:
                    
                    Qt = i
                        #print('Hello')
                    break
                    #elif K <= 1 :
                    #   Qt = i                            
                    #    #print('Hello')
                    #    break  
                '''
                if i == (Qs - 9):
                    for i in range(Qs, Qs - 20 ,-1):
                        K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3])# + abs(High[i-3] - High[i-4])
                        #print(K1)
                        if K1 <= 1:
                            Qt = i
                            break
                '''
            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                    #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])
                        #if K >=5 :
                            #M = M + 1

                        #if M >= 5 :    
                    if K1 <= 2  and K2 <= 2: #and K3 <= 1:
                        Qt = i
                            #print('Hello')
                        break
                        #elif K <= 1 :
                        #   Qt = i                            
                        #    #print('Hello')
                        #    break  
            
            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                    #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])
                        #if K >=5 :
                            #M = M + 1

                        #if M >= 5 :    
                    if K1 <= 3  and K2 <= 2 :#and K3 <= 2:
                        Qt = i
                            #print('Hello')
                        break
                        #elif K <= 1 :
                        #   Qt = i                            
                        #    #print('Hello')
                        #    break  
            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                    #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])

                    if K1 <= 3  and K2 <= 3 :#and K3 <= 2:
                        Qt = i
                            
                        break
            
            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                    #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])

                    if K1 <= 4  and K2 <= 3 :#and K3 <= 2:
                        Qt = i
                            
                        break

            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                    #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])

                    if K1 <= 4  and K2 <= 4 :#and K3 <= 2:
                        Qt = i
                            
                        break
            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                    #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])

                    if K1 <= 5  and K2 <= 4 :#and K3 <= 2:
                        Qt = i
                            
                        break
            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                    #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])

                    if K1 <= 5  and K2 <= 5 :#and K3 <= 2:
                        Qt = i
                            
                        break       
            if Qt == 0:
                Qt = Qs - 3

            K1 = 0 
            K2 = 0
            K3 = 0 
        
        #up type

        else:  
            #print('up')
            uptype = 1
            #R_index = 0
            #print(torch.where(High==119))
            #print()
            if name[k] == 'II':
                II = 1
            if name[k] == 'III':
                III = 1
            if name[k] == 'V1':
                V1 = 1
            if name[k] == 'V2':
                V2 = 1
            if name[k] == 'V3':
                V3 = 1 
            if name[k] == 'V4':
                V4 = 1 
            if name[k] == 'V5':
                V5 = 1 
            if name[k] == 'V6':
                V6 = 1 

            if k > 6 : 
                continue#break
            
        
            
            
            #elif B >= 4:
                #print(High_extreme)
                #High_extreme = High_extreme.sort()
                #print(High_extreme)
                #print(High)
            #    R_index1 = int(torch.where(High==torch.max(High))[0][2]) #High_extreme[2]
            #    R_index2 = int(torch.where(High==torch.max(High))[0][3]) #High_extreme[3]
                #print(torch.where(High==1))
        

            R_index = int(torch.max(High,0)[1]) 
            #print(torch.where(High==torch.max(High)))
            while R_index <= 50 or R_index >=200:
                #print("R_index"+str(R_index))
                #print(int(torch.min(High,0)[1]))
                #print(High[R_index]) 
                High[R_index] = torch.tensor([A])
                New_index = int(torch.max(High,0)[1]) 
                #print(New_index)
                R_index = New_index
            R_index1 = R_index - 2
            R_index2 = R_index + 2
            if torch.max(High)>=668:
                #R_index1 = R_index - 6
                R_index2 = R_index + 6###########!!!!!!change 2 to 6
            if R_index1 - 20 >= 0 :
                downrange = R_index1 - 20
            else:
                downrange = 0
            if R_index2 + 20 <= 239 :
                uprange = R_index2 + 20
            else:
                uprange = 239
            base = 0
            base = int(torch.mode(High[R_index1-40:R_index2+20])[0]) 
            #print(Base)
            if base == 0:
                #Base = torch.bincount(High)###!!!!!change  ###!!!!!change the range
                base = int(torch.mode(High)[0])
            
            
            High = high[k]#.cuda()
            for i in range(2,237):
                if (abs(High[i+1]-High[i]) + abs(High[i-1]-High[i])) >= 100:
                    High[i] = (High[i+1] + High[i-1])/2
            
            
            '''''''''''''''
            up
            Find S extreme
            '''''''''''''''
            
            S_index = 0
            #print(R_index2)
            for i in range(R_index2, uprange):
                if (i + 2 <= 239) and (i - 2 >= 0 ):
                    if (High[i-2]>High[i]) and (High[i+2]>High[i]):
                        S_index = i
                        break
                    elif ((High[i-2]>High[i]) and (High[i+2]>=High[i])) or ((High[i-2]>=High[i]) and (High[i+2]>High[i])):
                        S_index = i
                        break
                    elif (High[i-2]>=High[i]) and (High[i+2]>=High[i]):
                        S_index = i
                        break
                    elif ((High[i-2]>High[i]) and (High[i+1]>High[i])) or ((High[i-1]>High[i]) and (High[i+2]>High[i])):
                        S_index = i
                        break
                    elif (((High[i-2]>=High[i]) and (High[i+1]>High[i])) or ((High[i-2]>High[i]) and (High[i+1]>=High[i]))) or (((High[i-1]>=High[i]) and (High[i+2]>High[i])) or ((High[i-1]>High[i]) and (High[i+2]>=High[i]))):
                        S_index = i
                        break
                    elif (((High[i-2]>=High[i]) and (High[i+1]>=High[i])) or ((High[i-1]>=High[i]) and (High[i+2]>=High[i]))):
                        S_index = i
                        break
                    elif ((High[i-1]>High[i]) and (High[i+1]>High[i])):
                        S_index = i
                        break
                    elif ((High[i-1]>=High[i]) and (High[i+1]>High[i])) or ((High[i-1]>High[i]) and (High[i+1]>=High[i])):
                        S_index = i
                        break
                    elif ((High[i-1]>=High[i]) and (High[i+1]>=High[i])):
                        S_index = i
                        break
                elif (i + 1 <= 239) and (i - 1 >= 0 ):
                    if ((High[i-1]>High[i]) and (High[i+1]>High[i])):
                        S_index = i
                        break
                    elif ((High[i-1]>=High[i]) and (High[i+1]>High[i])) or ((High[i-1]>High[i]) and (High[i+1]>=High[i])):
                        S_index = i
                        break
                    elif ((High[i-1]>=High[i]) and (High[i+1]>=High[i])):
                        S_index = i
                        break

            if S_index == 0:
                S_index = R_index2 + 3
            
            #print(S_index)
            #if len(S_index)>3:
            #    S = int(torch.min(S_index))
            #else:
            #    S = int(torch.median(S_index))

        
            '''''''''''''''
            up
            Find S_final
            '''''''''''''''
            S_final = 0
            for i in range(S_index , S_index + 15 ):
                #print("i = "+ str(i))
                F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2])
                #print(F)
                if (F <= 1) and ( abs(base - High[i]) <= 10 ):
                    S_final = i
                    #print('Hello again')
                    #print(S_final)
                    break
            
            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2]) 
                    #print(F)
                    if (F <= 2) and ( abs(base - High[i]) <= 10 ):
                        S_final = i
                        break

            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2]) 
                    #print(F)
                    if (F <= 1) and ( abs(base - High[i]) <= 20 ):
                        S_final = i
                        break

            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2]) 
                    #print(F)
                    if (F <= 2) and ( abs(base - High[i]) <= 20 ):
                        S_final = i
                        break

            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2]) 
                    #print(F)
                    if (F <= 3) and ( abs(base - High[i]) <= 20 ):
                        S_final = i
                        break


            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    F = abs(High[i+1] - High[i]) + abs(High[i+2] - High[i+1]) + abs(High[i+3] - High[i+2]) 
                    #print(F)
                    if (F <= 2) and ( abs(base - High[i]) <= 30 ):
                        S_final = i
                        break      
        
            if S_final == 0:
                for i in range(S_index, S_index + 15 ):
                    if F <= base:
                        S_final = i
                        break
                    #print(S_final)
                    #print('Hello again')
                    
            if S_final == 0:
                S_final = S_index + 5
                for i in range(S_index, S_index + 15):
                    if High[i] < High[S_final]:
                        S_final = i
                    
            if S_final == 0:
                S_final = S_index + 3    

            '''''''''''''''
            up
            Find Q extreme
            '''''''''''''''

            Q_index = 0
            #print(R_index1)
            for i in range(R_index1, downrange, -1):

                if (i + 2 <= 239) and (i - 2 >= 0 ):
                    #print(High[i])
                    #print(abs(High[i] - High[S_final]))
                    #print(str(S))
                    #print(str(i) + ',' + str(abs(High[i] - High[S])))
                    if (abs(High[i] - base) < 20):
                        
                        if (High[i-2]>High[i]) and (High[i+2]>High[i]):
                            Q_index = i
                            break
                        elif ((High[i-2]>High[i]) and (High[i+2]>=High[i])) or ((High[i-2]>=High[i]) and (High[i+2]>High[i])):
                            Q_index = i
                            break
                        elif (High[i-2]>=High[i]) and (High[i+2]>=High[i]):
                            Q_index = i
                            break
                        elif ((High[i-2]>High[i]) and (High[i+1]>High[i])) or ((High[i-1]>High[i]) and (High[i+2]>High[i])):
                            Q_index = i
                            break
                        elif (((High[i-2]>=High[i]) and (High[i+1]>High[i])) or ((High[i-2]>High[i]) and (High[i+1]>=High[i]))) or (((High[i-1]>=High[i]) and (High[i+2]>High[i])) or ((High[i-1]>High[i]) and (High[i+2]>=High[i]))):
                            Q_index = i
                            break
                        elif (((High[i-2]>=High[i]) and (High[i+1]>=High[i])) or ((High[i-1]>=High[i]) and (High[i+2]>=High[i]))):
                            Q_index = i
                            break
                        elif ((High[i-1]>High[i]) and (High[i+1]>High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]>=High[i]) and (High[i+1]>High[i])) or ((High[i-1]>High[i]) and (High[i+1]>=High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]>=High[i]) and (High[i+1]>=High[i])):
                            Q_index = i
                            break
                elif (i + 1 <= 239) and (i - 1 >= 0 ):
                    if (abs(High[i] - base) < 20):
                        if ((High[i-1]>High[i]) and (High[i+1]>High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]>=High[i]) and (High[i+1]>High[i])) or ((High[i-1]>High[i]) and (High[i+1]>=High[i])):
                            Q_index = i
                            break
                        elif ((High[i-1]>=High[i]) and (High[i+1]>=High[i])):
                            Q_index = i
                            break

            if Q_index == 0:
                for i in range(R_index1, downrange, -1):
                    if (i + 2 <= 239) and (i - 2 >= 0 ):
                        #print(str(S))
                        #print(str(i) + ',' + str(abs(High[i] - High[S])))
                        if (abs(High[i] - base) < 30):
                            if (High[i-2]>High[i]) and (High[i+2]>High[i]):
                                Q_index = i
                                break
                            elif ((High[i-2]>High[i]) and (High[i+2]>=High[i])) or ((High[i-2]>=High[i]) and (High[i+2]>High[i])):
                                Q_index = i
                                break
                            elif (High[i-2]>=High[i]) and (High[i+2]>=High[i]):
                                Q_index = i
                                break
                            elif ((High[i-2]>High[i]) and (High[i+1]>High[i])) or ((High[i-1]>High[i]) and (High[i+2]>High[i])):
                                Q_index = i
                                break
                            elif (((High[i-2]>=High[i]) and (High[i+1]>High[i])) or ((High[i-2]>High[i]) and (High[i+1]>=High[i]))) or (((High[i-1]>=High[i]) and (High[i+2]>High[i])) or ((High[i-1]>High[i]) and (High[i+2]>=High[i]))):
                                Q_index = i
                                break
                            elif (((High[i-2]>=High[i]) and (High[i+1]>=High[i])) or ((High[i-1]>=High[i]) and (High[i+2]>=High[i]))):
                                Q_index = i
                                break
                            
                            elif ((High[i-1]>High[i]) and (High[i+1]>High[i])):
                                Q_index = i
                                break
                            elif ((High[i-1]>=High[i]) and (High[i+1]>High[i])) or ((High[i-1]>High[i]) and (High[i+1]>=High[i])):
                                Q_index = i
                                break
                            elif ((High[i-1]>=High[i]) and (High[i+1]>=High[i])):
                                Q_index = i
                                break
                    elif (i + 1 <= 239) and (i - 1 >= 0 ):
                        if (abs(High[i] - base) < 30): ##############High[S_index]
                            if ((High[i-1]>High[i]) and (High[i+1]>High[i])):
                                Q_index = i
                                break
                            elif ((High[i-1]>=High[i]) and (High[i+1]>High[i])) or ((High[i-1]>High[i]) and (High[i+1]>=High[i])):
                                Q_index = i
                                break
                            elif ((High[i-1]>=High[i]) and (High[i+1]>=High[i])):
                                Q_index = i
                                break
            if Q_index == 0:
                Q_index = R_index1 - 3

            '''''''''''''''
            up
            Find Q_final
            '''''''''''''''
            Q_final = 0
            D = 0
            for i in range(Q_index , Q_index - 15 ,-1):
                D = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3])
                if High[i] >= base or D <= 1:
                    Q_final = i
                    break
                '''
                D = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3])
                if (D <= 1) and ( abs(High[S_final] - High[i]) <= 20 ):
                    Q_final = i
                    #print('Hello')
                    break
                elif (D <= 2) and ( abs(High[S_final] - High[i]) <= 20 ):                        
                    Q_final = i
                    #print('Hello')
                    break   
                '''

            if Q_final == 0:
                Q_final = Q_index - 3

            '''''''''''''''
            up
            Find Qs
            '''''''''''''''
            
            #find QS extreme
            R = 0
            num1 = 0
            num2 = 0
            up = 0
            Qs = 0
            for i in range(Q_final , Q_final - 20 ,-1):
                R = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3])
                #print (R)
                #if R >=3 :
                    #num = num + 1
                #print (num)
                if name[k] == 'aVR':
                    if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0): #and (High[i-3]-High[i]<0 or High[i-2]-High[i-1]<=0):
                        num1 = num1 + 1
                    if num1 >=3:
                        up = 1
                    if High[i-1]-High[i]>0 and up == 1:
                    #if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0) and up==1:
                        num2 = num2 + 1
                    if num2 >=3:
                        Qs = i  ######changed from Qs = i + 3
                        break
                else :
                    if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) :#and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0):
                        num1 = num1 + 1
                    if num1 >=3:
                        up = 1
                        #print(i)
                    if High[i-1]-High[i]<0 and up == 1:
                    #if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0) and up==1:
                        num2 = num2 + 1
                    if num2 >=3:
                        Qs = i  ######changed from Qs = i + 3
                        break

            if Qs == 0:
                for i in range(Q_final , Q_final - 20 ,-1):
                    if name[k] == 'aVR':
                        if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0): #and (High[i-3]-High[i]<0 or High[i-2]-High[i-1]<=0):
                            num1 = num1 + 1
                        if num1 >=2:
                            up = 1
                        if High[i-1]-High[i]>0 and up == 1:
                        #if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0) and up==1:
                            num2 = num2 + 1
                        if num2 >=3:
                            Qs = i  ######changed from Qs = i + 3
                            break
                    else :
                        if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) :#and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0):
                            num1 = num1 + 1
                        if num1 >=2:
                            up = 1
                            #print(i)
                        if High[i-1]-High[i]<0 and up == 1:
                        #if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0) and up==1:
                            num2 = num2 + 1
                        if num2 >=3:
                            Qs = i  ######changed from Qs = i + 3
                            break

            if Qs == 0:
                for i in range(Q_final , Q_final - 20 ,-1):
                    if name[k] == 'aVR':
                        if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0): #and (High[i-3]-High[i]<0 or High[i-2]-High[i-1]<=0):
                            num1 = num1 + 1
                        if num1 >=3:
                            up = 1
                        if High[i-1]-High[i]>0 and up == 1:
                        #if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0) and up==1:
                            num2 = num2 + 1
                        if num2 >=2:
                            Qs = i  ######changed from Qs = i + 3
                            break
                    else :
                        if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) :#and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0):
                            num1 = num1 + 1
                        if num1 >=3:
                            up = 1
                            #print(i)
                        if High[i-1]-High[i]<0 and up == 1:
                        #if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0) and up==1:
                            num2 = num2 + 1
                        if num2 >=2:
                            Qs = i  ######changed from Qs = i + 3
                            break

            if Qs == 0:
                for i in range(Q_final , Q_final - 20 ,-1):
                    if name[k] == 'aVR':
                        if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0): #and (High[i-3]-High[i]<0 or High[i-2]-High[i-1]<=0):
                            num1 = num1 + 1
                        if num1 >=2:
                            up = 1
                        if High[i-1]-High[i]>0 and up == 1:
                        #if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0) and up==1:
                            num2 = num2 + 1
                        if num2 >=2:
                            Qs = i  ######changed from Qs = i + 3
                            break
                    else :
                        if (High[i-2]-High[i]>0 or High[i-1]-High[i]>=0) :#and (High[i-3]-High[i-1]>0 or High[i-2]-High[i-1]>=0):
                            num1 = num1 + 1
                        if num1 >=2:
                            up = 1
                            #print(i)
                        if High[i-1]-High[i]<0 and up == 1:
                        #if (High[i-2]-High[i]<0 or High[i-1]-High[i]<=0) and (High[i-3]-High[i-1]<0 or High[i-2]-High[i-1]<=0) and up==1:
                            num2 = num2 + 1
                        if num2 >=2:
                            Qs = i  ######changed from Qs = i + 3
                            break

            if Qs == 0:
                Qs = Q_final - 3

            
                         
            num1 = 0
            num2 = 0
            up = 0
            #print(Qs)
            #Qs = int(torch.median(Qs))

            '''''''''''''''
            up
            Find Qt
            '''''''''''''''   

            K1 = 0 
            K2 = 0
            K3 = 0
            M = 0
            Qt = 0
            '''
            if Q_final < Qs:
                Qt = Q_final
                Q_final = Q_index
            '''
            #else: 
            for i in range(Qs , Qs - 10 ,-1):
                K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])
                #print("K1:"+str(K1))
                #print("K2:"+str(K2))
                #print("K3:"+str(K3))
                    #if K >=5 :
                        #M = M + 1

                    #if M >= 5 :    
                if K1 <= 2  and K2 <= 1 :#and K3 <= 1:
                    Qt = i
                        #print('Hello')
                    break
                    #elif K <= 1 :
                    #    Qt = i                            
                    #    #print('Hello')
                
                '''    
                #    break    
                if i == (Qs - 9):
                    for i in range(Qs, Qs - 20 ,-1):
                        K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3])# + abs(High[i-3] - High[i-4])
                        #print(K1)
                        if K1 <= 1:
                            Qt = i
                            break
                '''
            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                    #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])
                        #if K >=5 :
                            #M = M + 1

                        #if M >= 5 :    
                    if K1 <= 2  and K2 <= 2 :#and K3 <= 1:
                        Qt = i
                            #print('Hello')
                        break
                        #elif K <= 1 :
                        #   Qt = i                            
                        #    #print('Hello')
                        #    break  
            
            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                    #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])
                        #if K >=5 :
                            #M = M + 1

                        #if M >= 5 :    
                    if K1 <= 3  and K2 <= 2 :#and K3 <= 2:
                        Qt = i
                            #print('Hello')
                        break
                        #elif K <= 1 :
                        #   Qt = i                            
                        #    #print('Hello')
                        #    break  
            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
                    #K3 = abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5]) + abs(High[i-5] - High[i-6])
                        #if K >=5 :
                            #M = M + 1

                        #if M >= 5 :    
                    if K1 <= 3  and K2 <= 3:# and K3 <= 2:
                        Qt = i
                            #print('Hello')
                        break
                        #elif K <= 1 :
                        #   Qt = i                            
                        #    #print('Hello')
            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
        
                    if K1 <= 4  and K2 <= 3:# and K3 <= 2:
                        Qt = i
                        break

            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
        
                    if K1 <= 4  and K2 <= 4:# and K3 <= 2:
                        Qt = i
                        break

            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
        
                    if K1 <= 5  and K2 <= 4:# and K3 <= 2:
                        Qt = i
                        break


            if Qt == 0:
                for i in range(Qs, Qs - 10 ,-1):
                    K1 = abs(High[i] - High[i-1]) + abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) #+ abs(High[i-4] - High[i-5])
                    K2 = abs(High[i-1] - High[i-2]) + abs(High[i-2] - High[i-3]) + abs(High[i-3] - High[i-4]) + abs(High[i-4] - High[i-5])
        
                    if K1 <= 5  and K2 <= 5:# and K3 <= 2:
                        Qt = i
                        break
            if Qt == 0:
                Qt = Qs - 3
            
            M = 0
            K1 = 0
            K2 = 0
            K3 = 0
        
        ###############################
        ######## new feature ##########
        ###############################
        '''
        if Qs <= 0 :#or Q_final == 0: or S_final == 0:
            for i in range(Q_final , S_final):
                area = area + (High[i] - High[S_final])
            
            square =+ 1
            print("Qt: "+str(Qt)+"; Q: "+str(Q_final)+"; S: "+str(S_final))
            print("Q_index: "+str(Q_index)+"; S_index: "+str(S_index))
        
            if k == 0:
                I_max = torch.max(High[Q_final:S_final]) - High[S_final]
            if k == 6:
                PvN = (torch.max(High[Q_final:S_final]) - High[S_final]) / (High[S_final] - torch.min(High[Q_final:S_final]))
                V1_max = torch.max(High[Q_final:S_final]) - High[S_final]
                
            ###################################################
            if k == 4:
                aVL_max = torch.max(High[Q_final:S_final])  - High[S_final]
                
            

            
            print("Q: "+str(Q_final)+"; S: "+str(S_final))
            T1 = S_final - Q_final
            print("T1: "+str(T1))
            if T1 >= 20 :
                print('The picture ' + name[k] + ' has something wrong with QS distance.')
                Positive = 1
            
        '''
        
        if k < 6:
            
            #print("Qt: "+str(Qt)+"; Q: "+str(Q_final)+"; S: "+str(S_final))
            #print("Q_index: "+str(Q_index)+"; S_index: "+str(S_index))
            T1[k] = S_final - Q_final
            T2[k] = Q_final - Qt

            T1_sum = T1_sum + T1
            T2_sum = T2_sum + T2
            curve = curve + 1 
            if T1[k] >= 15:
                #print('The picture ' + name[k] + ' has something wrong with QS distance and platform distance.')
                Positive1 = Positive1 + 1
            if T2[k] <= 15:
                Positive2 = Positive2 + 1
            '''    
            print("T1: "+str(T1))
            print("T2: "+str(T2))
            '''
            for i in range(Q_final , S_final):
                area = area + abs(High[i] - High[S_final])
             

        if k == 0:
            
            I_max = torch.max(High[Q_final:S_final]) - High[S_final]
        
        if k == 6:
            '''
            print("Qt: "+str(Qt)+"; Q: "+str(Q_final)+"; S: "+str(S_final))
            print("Q_index: "+str(Q_index)+"; S_index: "+str(S_index))
            
            print("torch.max(High[Q_final:S_final])"+str(torch.max(High[Q_final:S_final])))
            print("High[S_final]"+ str(High[S_final]))
            print("##########"+str(torch.max(High[Qt:S_final])))
            print("##########"+str(torch.max(High[Qt:S_final])))
            '''
            V1_max = torch.max(High[Qt:S_final]) - High[S_final]
            X = abs(torch.max(High[Qt:S_final])- High[S_final])
            Y = abs(High[S_final] - torch.min(High[Qt:S_final]))
            #print(X)
            #print(Y)
            if int(Y) == 0:
                Y = 1
            PvN = int(X)/int(Y)
        if k == 4:
            
            aVL_max = torch.max(High[Q_final:S_final]) - High[S_final]
    
    result = torch.zeros(25)
    result[0] = T1[0]
    result[1] = T1[1]
    result[2] = T1[2]
    result[3] = T1[3]
    result[4] = T1[4]
    result[5] = T1[5]
    result[6] = T2[0]
    result[7] = T2[1]
    result[8] = T2[2]
    result[9] = T2[3]
    result[10] = T2[4]
    result[11] = T2[5]
    result[12] = Positive1 
    result[13] = Positive2
    result[14] = PvN
    result[15] = I_minus
    result[16] = II_minus
    result[17] = III_minus
    result[18] = area
    result[19] = V1
    result[20] = V2
    result[21] = V3
    result[22] = V4
    result[23] = V5
    result[24] = V6

    return result



EPOCH = 100   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 10      #批处理尺寸(batch_size)
LR = 0.001        #学习率

data_transform = transforms.Compose([
    #transforms.Resize((120,240)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


parser = argparse.ArgumentParser(description='PyTorch classification Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/network.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()


train_dataset = datasets.ImageFolder(root='./new_train/',transform=data_transform)
train_dataloader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=BATCH_SIZE)

test_dataset = datasets.ImageFolder(root='./T/',transform=data_transform)
test_dataloader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=BATCH_SIZE)

train_dataset_size = len(train_dataset)
class_name= train_dataset.classes
classes = ('A','B','None')

best_acc = 85
model = Network().cuda()
cast = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-8)



if __name__ == '__main__':
    #Features = torch.FloatTensor(25).zero_()
    #Curve = torch.FloatTensor(12,240).zero_()
    for i, data in enumerate(train_dataloader, 0):   
        length = len(train_dataloader)
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        #print(inputs.size())
        #print(labels.size())
        features1 = torch.FloatTensor(len(inputs[:,0,0,0]),25).zero_()
        target1 = torch.LongTensor(len(inputs[:,0,0,0])).zero_()
        curve1 = torch.FloatTensor(len(inputs[:,0,0,0]),12,240).zero_()
        for k in range(len(inputs[:,0,0,0])):    
            im = inputs[k]
            target1[k] = labels[k]
            #save_image(im,'ECG.png')
            #im = im.permute(2,0,1)
            #save_image(im,'ECG.png')
            #im = (im[0] + im[1] +im[2])/3
            #print(im.size())
            im[im > 100/255] = 255
            im[im != 255 ] = 0
            im = (im[0] + im[1] + im[2])/3
            #sio.savemat("line1.mat",{"line1":im.numpy()})
            #ECG  = torch.cat((torch.cat((im,im)),im))
            #save_image(ECG,'ECG.png')
            x = torch.IntTensor(12).zero_()
            y = torch.IntTensor(12).zero_()
            w = 240
            h = torch.zeros(12)
            high = torch.zeros((12,240))
            name = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
            high[0], high[1],high[2] = ExtractPixel(im,165 , 165 + 240)
            high[3], high[4],high[5] = ExtractPixel(im,165 + 300, 165 + 300 + 240)
            high[6], high[7],high[8] = ExtractPixel(im,165 + 300 * 2, 165 + 300 * 2 + 240)
            high[9], high[10],high[11] = ExtractPixel(im,165 + 300 * 3, 165 + 300 * 3 + 240)
            '''
            sio.savemat("line0.mat",{"line0":high[0].numpy()})
            sio.savemat("line1.mat",{"line1":high[1].numpy()})
            sio.savemat("line2.mat",{"line2":high[2].numpy()})
            sio.savemat("line3.mat",{"line3":high[3].numpy()})
            sio.savemat("line4.mat",{"line4":high[4].numpy()})
            sio.savemat("line5.mat",{"line5":high[5].numpy()})
            sio.savemat("line6.mat",{"line6":high[6].numpy()})
            sio.savemat("line7.mat",{"line7":high[7].numpy()})
            sio.savemat("line8.mat",{"line8":high[8].numpy()})
            sio.savemat("line9.mat",{"line9":high[9].numpy()})
            sio.savemat("line10.mat",{"line10":high[10].numpy()})
            sio.savemat("line11.mat",{"line11":high[11].numpy()})
            '''


            V1_down = 0
            V2_down = 0
            V5_up = 0
            V6_up = 0
            V_count_up = 0
            V_count_down = 0
            Positive = 0
            curve1[k] = high
            features1[k] = ExtractFeatures(high,name)
        if i == 0 :
            Features1 = features1
            Classification1 = target1
            Curve1 = curve1
            #print(Features)
            #print(Features1.size())
            #print(Classification)
            #print(Classification1.size())
            #print(Curve1.size())
        else:
            Features1 = torch.cat((Features1,features1))
            Curve1 = torch.cat((Curve1,curve1))
            Classification1 = torch.cat((Classification1,target1))
            #print(Features)
            #print(Features1.size())
            #print(Classification)
            #print(Classification1.size())
            #print(Curve1.size())

    torch_dataset1 = Data.TensorDataset(Features1, Classification1)
        #print(Features.size())
        #print(Classification.size())
    loader1 = DataLoader(
        dataset=torch_dataset1,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
    )




    for i, data in enumerate(test_dataloader, 0):   
        length = len(test_dataloader)
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        #print(inputs.size())
        #print(labels.size())
        features2 = torch.FloatTensor(len(inputs[:,0,0,0]),25).zero_()
        target2 = torch.LongTensor(len(inputs[:,0,0,0])).zero_()
        curve2 = torch.FloatTensor(len(inputs[:,0,0,0]),12,240).zero_()
        for k in range(len(inputs[:,0,0,0])):    
            im = inputs[k]
            target2[k] = labels[k]
            #save_image(im,'ECG.png')
            #im = im.permute(2,0,1)
            #save_image(im,'ECG.png')
            #im = (im[0] + im[1] +im[2])/3
            #print(im.size())
            im[im > 100/255] = 255
            im[im != 255 ] = 0
            im = (im[0] + im[1] + im[2])/3
            #sio.savemat("line1.mat",{"line1":im.numpy()})
            #ECG  = torch.cat((torch.cat((im,im)),im))
            #save_image(ECG,'ECG.png')
            x = torch.IntTensor(12).zero_()
            y = torch.IntTensor(12).zero_()
            w = 240
            h = torch.zeros(12)
            high = torch.zeros((12,240))
            name = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
            high[0], high[1],high[2] = ExtractPixel(im,165 , 165 + 240)
            high[3], high[4],high[5] = ExtractPixel(im,165 + 300, 165 + 300 + 240)
            high[6], high[7],high[8] = ExtractPixel(im,165 + 300 * 2, 165 + 300 * 2 + 240)
            high[9], high[10],high[11] = ExtractPixel(im,165 + 300 * 3, 165 + 300 * 3 + 240)
            '''
            sio.savemat("line0.mat",{"line0":high[0].numpy()})
            sio.savemat("line1.mat",{"line1":high[1].numpy()})
            sio.savemat("line2.mat",{"line2":high[2].numpy()})
            sio.savemat("line3.mat",{"line3":high[3].numpy()})
            sio.savemat("line4.mat",{"line4":high[4].numpy()})
            sio.savemat("line5.mat",{"line5":high[5].numpy()})
            sio.savemat("line6.mat",{"line6":high[6].numpy()})
            sio.savemat("line7.mat",{"line7":high[7].numpy()})
            sio.savemat("line8.mat",{"line8":high[8].numpy()})
            sio.savemat("line9.mat",{"line9":high[9].numpy()})
            sio.savemat("line10.mat",{"line10":high[10].numpy()})
            sio.savemat("line11.mat",{"line11":high[11].numpy()})
            '''


            V1_down = 0
            V2_down = 0
            V5_up = 0
            V6_up = 0
            V_count_up = 0
            V_count_down = 0
            Positive = 0
            curve2[k] = high
            features2[k] = ExtractFeatures(high,name)
        if i == 0 :
            Features2 = features2
            Classification2 = target2
            Curve2 = curve2
            #print(Features)
            #print(Features2.size())
            #print(Classification)
            #print(Classification2.size())
            #print(Curve2.size())
        else:
            Features2 = torch.cat((Features2,features2))
            Curve2 = torch.cat((Curve2,curve2))
            Classification2 = torch.cat((Classification2,target2))
            #print(Features)
            #print(Features2.size())
            #print(Classification)
            #print(Classification2.size())
            #print(Curve2.size())




    torch_dataset2 = Data.TensorDataset(Features2, Classification2)
        #print(Features.size())
        #print(Classification.size())
    loader2 = DataLoader(
        dataset=torch_dataset2,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
    )

        #print(result)
    #Classification = torch.unsqueeze(Classification,1)

    print("Start Training, Network!")  # 定义遍历数据集的次数
    with open("acc(own_net).txt", "w") as f:
        with open("log(own_net).txt", "w")as f2:            
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                model.train(True)
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                #for i in range(len(Features[:,0])):
                for i, data in enumerate(loader1,0):    
                    # 准备数据
                    #print(i)
                    
                    inputs, labels = data
                    inputs, labels = inputs.cuda(), labels.cuda()
                    #inputs = torch.unsqueeze(inputs,0)
                    #labels = torch.unsqueeze(labels,0)
                    #features = torch.FloatTensor(len(inputs[:,0,0,0]),25).zero_()#.cuda()
                    #print(len(inputs[:,0,0,0]))
                    #if i == 0 :
                    #    print(inputs.size())
                    #    print(labels.size())
                    
                    outputs = model(inputs).cuda()

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
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
                
                
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for i, data in enumerate(loader2,0):   
                        model.eval() 
                        # 准备数据
                        #print(i)
                    
                        inputs, labels = data
                        inputs, labels = inputs.cuda(), labels.cuda()
                        outputs = model(inputs).cuda()
                        #outputs = model(inputs).cuda()
                        # 取得分最高的那个类 (outputs.data的索引号)
                        #labels = labels.squeeze()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(model.state_dict(), '%s/model_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc(18).txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc

            print("Training Finished, TotalEPOCH=%d" % EPOCH)
    