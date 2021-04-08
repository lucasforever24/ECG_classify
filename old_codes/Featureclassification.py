import glob
import os
from PIL import Image
import numpy as np
import scipy.io as sio
from sklearn.svm import SVC




def feature(im):
    #im = np.array(Image.open("2.pic_hd.jpg").convert('L'))
    im[ im > 100 ] = 255
    im[ im != 255 ] = 0
    #im = Image.fromarray(im)
    #im.save('7_pic.png')


    # 图片的宽度和高度
    img_size = im.size
    #print("图片宽度和高度分别是{}".format(img_size))
    '''
    裁剪：传入一个元组作为参数
    元组里的元素分别是：（距离图片左边界距离x， 距离图片上边界距离y，距离图片左边界距离+裁剪框宽度x+w，距离图片上边界距离+裁剪框高度y+h）
    '''
    # 截取图片中一块宽是200和高是120的
    M = 1#int(input("Please input the printing type (1 or 2): "))
    x = np.zeros(shape = (12))
    y = np.zeros(shape = (12))
    w = 240
    h = np.zeros(shape = (12))
    high = np.zeros(shape = (12,240),dtype = int)
    #region = np.zeros(shape = (12,80,240))
    #district = np.zeros(shape = (6,80,240))
    name = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']


    if M == 1:

        #=================宽度改变================
        ECG = im
        line = 0
        line_extreme_up = 0
        #line_extreme_down = 0
        before_count_up = 0
        no_area = 0
        #before_count_down = 0
        line1_extreme = []
        line3_extreme = []
        line_candidate1 = []
        line_candidate2 = []
        line_candidate3 = []
        line1 = np.zeros(shape = 240,dtype=int)
        line2 = np.zeros(shape = 240,dtype=int)
        line3 = np.zeros(shape = 240,dtype=int)
        tem_counts1 = 0
        tem_counts2 = 0
        tem_counts3 = 0

        ############################第一列###########################

        for j in range(165  ,165 + 240):
            for i  in range (181,670):
                if ECG[i][j] == 0 and line == 0:
                    
                    #print(str(j)+ " " + str(i))
                    line_candidate1.append(669 - i)
                    #print(line_candidate1)
                    if i == 181:
                        if before_count_up == 0:
                            line_extreme_up = 1
                            line1_extreme.append(j)
                        else :
                            line_extreme_up = 1
                            before_count_up = 1    

                elif (i >= 182 ) and (ECG[i][j] != 0) and (ECG[i-1][j] == 0) and (line == 0) :
                    line = line + 1
                    continue
                if ECG[i][j] == 0 and line == 1:
                    line_candidate2.append(669 - i)
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
                    line_candidate3.append(669 - i)
                    '''
                    if len(line_candidate3) < 1:
                        line_candidate3.append(669 - i) 
                    elif (line_candidate3[len(line_candidate3)-1] - (669 - i)) == 1:
                        line_candidate3.append(669 - i)
                    '''
                elif (ECG[i][j] != 0) and (ECG[i-1][j] == 0) and (line == 2):
                    line = 0
                    break


            x = j - 165
            #print(x)
            #print(line_candidate1)
            #print(line_candidate2)
            #print(line_candidate3)

            tem_counts1 = np.bincount(line1)
            tem_counts2 = np.bincount(line2)
            tem_counts3 = np.bincount(line3)
                #返回众数
            tem_counts1 = np.argmax(tem_counts1)
            tem_counts2 = np.argmax(tem_counts2)
            tem_counts3 = np.argmax(tem_counts3)

            if line_candidate2 == []:
                line_candidate2 = line_candidate1
                line_candidate3 = line_candidate1

            elif line_candidate3 == []:
                line_candidate3 = line_candidate2

            
            if ((x == 0) or (x == 1)) :

                if (line_extreme_up == 1 and before_count_up == 0): #and (begin == 0) :
                    line1[x] = np.max(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)
                    before_count_up = 1
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                    #begin = 1
                    
                    
                elif (line_extreme_up == 0 and before_count_up == 1): #and (begin == 1):
                    line1[x] = 669 - 181
                    line2[x] = np.min(line_candidate1)
                    line3[x] = np.min(line_candidate2)
                    before_count_up = 1
                    before_count_down = 0
                    line_extreme_up = 0
                    line_extreme_down = 0


                elif (line_extreme_up == 1 and before_count_up == 1): #and (begin == 1) :
                    line1[x] = np.max(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)
                    before_count_up = 0
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                    #begin = 0


                elif (line_extreme_up == 0 and before_count_up == 0):
                    line1[x] = np.min(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)


            else:
                if (line_extreme_up == 1 and before_count_up == 0):
                    line1[x] = 669 - 181
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate2)
                        else :
                            line2[x] = np.min(line_candidate2)                
                    
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate3)
                    elif (line3[x-2]-line3[x-1] > 0):   
                        line3[x] = np.min(line_candidate3)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate3)
                        else :
                            line3[x] = np.min(line_candidate3)


                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)
                        
                        #if line1[x] == (669 - 181 - 25) or 
                    if (line2[x-1] >= line2[x] + 120):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate1)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate1)
                            else :
                                line2[x] = np.min(line_candidate1)


                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)


                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)

                    before_count_up = 1
                    #before_count_down = 1
                    line_extreme_up = 0
                    #line_extreme_down = 0
                
                elif (line_extreme_up == 0 and before_count_up == 1):
                    line1[x] = 669 - 181
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate1)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate1)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate1)
                        else :
                            line2[x] = np.min(line_candidate1)

                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate2)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = np.min(line_candidate2)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate2)
                        else :
                            line3[x] = np.min(line_candidate2)


                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate1)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate1)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate1)
                            else :
                                line3[x] = np.min(line_candidate1)

                    '''  
                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)
                    '''

                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)


                    before_count_up = 1
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                
                elif (line_extreme_up == 1 and before_count_up == 1):
                    line1[x] = 669 - 181


                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate2)
                        else :
                            line2[x] = np.min(line_candidate2)


                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate3)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = np.min(line_candidate3)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate3)
                        else :
                            line3[x] = np.min(line_candidate3)
                    
                    
                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)

                        
                        #if line1[x] == (669 - 181 - 25) or 
                    if (line2[x-1] >= line2[x] + 120):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate1)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate1)
                            else :
                                line2[x] = np.min(line_candidate1)


                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)


                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)




                    line_extreme_up = 0
                    #line_extreme_down = 0
                    before_count_up = 0
                    #before_count_down = 1



                elif (line_extreme_up == 0 and before_count_up == 0):
                    if (line1[x-1] - line2[x-1] >=0 ) and (line1[x-1] - line2[x-1] < (len(line_candidate1) + 7)) and ((np.min(line_candidate1) - np.max(line_candidate2)) >= 50) : 
                        line1[x] = np.min(line_candidate1)
                        line2[x] = np.max(line_candidate1)
                        #print(line1[x-1] - line2[x-1])
                        #print(len(line_candidate1))
                        
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-3]-line3[x-2] > 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)   
                            
                        if (line3[x-1] >= line3[x] + 120 ):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate1)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate1)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate1)
                                else :
                                    line3[x] = np.min(line_candidate1)
                    

                        if (line3[x] == line2[x]):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate2)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate2)
                                else :
                                    line3[x] = np.min(line_candidate2)


                    elif (line2[x-1] - line3[x-1] >=0 ) and (line2[x-1] - line3[x-1] < (len(line_candidate2) + 7)) and ((np.min(line_candidate2) - np.max(line_candidate3)) >= 50) : 
                        #print(line2[x-1] - line3[x-1])
                        #print(len(line_candidate1))
                        if (line1[x-2]-line1[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line1[x] = np.max(line_candidate1)    
                        elif (line1[x-2]-line1[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line1[x] = np.min(line_candidate1)
                        elif (line1[x-2]-line1[x-1] == 0):
                            if (line1[x-3]-line1[x-2] < 0):
                                line1[x] = np.max(line_candidate1)
                            elif (line1[x-3]-line1[x-2] > 0):
                                line1[x] = np.min(line_candidate1)
                            else :
                                line1[x] = np.min(line_candidate1)
                        
                        line2[x] = np.min(line_candidate2)
                        line3[x] = np.max(line_candidate2)              
                    
                    else:
                        if (line1[x-2]-line1[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line1[x] = np.max(line_candidate1)
                            
                        elif (line1[x-2]-line1[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line1[x] = np.min(line_candidate1)
                        elif (line1[x-2]-line1[x-1] == 0):
                            if (line1[x-3]-line1[x-2] < 0):
                                line1[x] = np.max(line_candidate1)
                            elif (line1[x-3]-line1[x-2] > 0):
                                line1[x] = np.min(line_candidate1)
                            else :
                                line1[x] = np.min(line_candidate1)


                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.max(line_candidate2)
                            elif (line2[x-3]-line2[x-2] > 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)

                        
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.max(line_candidate3)
                            elif (line3[x-3]-line3[x-2] > 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)


                        #if (line2[x] == np.max(line_candidate1)) or  (line2[x] == np.min(line_candidate1)) or 
                        if (line3[x-1] >= line3[x] + 120 ):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate2)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate2)
                                else :
                                    line3[x] = np.min(line_candidate2)

                        
                        #if (line1[x] == 669 - 181) or 
                        if (line2[x-1] >= line2[x] + 120 ):
                            if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line2[x] = np.max(line_candidate1)
                            elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line2[x] = np.min(line_candidate1)
                            elif (line2[x-2]-line2[x-1] == 0):
                                if (line2[x-3]-line2[x-2] < 0):
                                    line2[x] = np.min(line_candidate1)
                                else :
                                    line2[x] = np.min(line_candidate1)

                        if (line2[x] == line1[x]):
                            if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line2[x] = np.max(line_candidate2)
                            elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line2[x] = np.min(line_candidate2)
                            elif (line2[x-2]-line2[x-1] == 0):
                                if (line2[x-3]-line2[x-2] < 0):
                                    line2[x] = np.min(line_candidate2)
                                else :
                                    line2[x] = np.min(line_candidate2)


                        if (line3[x] == line2[x]):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate3)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate3)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate3)
                                else :
                                    line3[x] = np.min(line_candidate3)



                    before_count_up = 0
                    before_count_down = 0
                    line_extreme_up = 0
                    line_extreme_down = 0
                
                
            line = 0
            tem_counts1 = 0
            tem_counts2 = 0
            tem_counts3 = 0
            line_candidate1 = []
            line_candidate2 = []
            line_candidate3 = []     
        #sio.savemat("line1.mat",{"line1":line1})
        #sio.savemat("line2.mat",{"line2":line2})
        #sio.savemat("line3.mat",{"line3":line3})
        high[0] = line1
        high[1] = line2
        high[2] = line3


        ##第二列##
        
        #ECG = im
        line = 0
        line_extreme_up = 0
        #line_extreme_down = 0
        before_count_up = 0
        no_area = 0
        #before_count_down = 0
        line1_extreme = []
        line3_extreme = []
        line_candidate1 = []
        line_candidate2 = []
        line_candidate3 = []
        line1 = np.zeros(shape = 240,dtype=int)
        line2 = np.zeros(shape = 240,dtype=int)
        line3 = np.zeros(shape = 240,dtype=int)
        tem_counts1 = 0
        tem_counts2 = 0
        tem_counts3 = 0

        for j in range(165 + 300  ,165 + 300 + 240):
            for i  in range (181,670):
                if ECG[i][j] == 0 and line == 0:
                    #print(str(j)+ " " + str(i))
                    line_candidate1.append(669 - i)
                    if i == 181:
                        if before_count_up == 0:
                            line_extreme_up = 1
                            line1_extreme.append(j)
                        else :
                            line_extreme_up = 1
                            before_count_up = 1


                    '''
                    if len(line_candidate1) < 1:
                        line_candidate1.append(635 - i) 
                    elif (line_candidate1[len(line_candidate1)-1] - (635 - i)) == 1:
                        line_candidate1.append(635 - i)
                    '''
                elif (i >= 182 ) and (ECG[i][j] != 0) and (ECG[i-1][j] == 0) and (line == 0) :
                    line = line + 1
                    continue
                if ECG[i][j] == 0 and line == 1:
                    line_candidate2.append(669 - i)
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
                    line_candidate3.append(669 - i)
                    '''
                    if len(line_candidate3) < 1:
                        line_candidate3.append(669 - i) 
                    elif (line_candidate3[len(line_candidate3)-1] - (669 - i)) == 1:
                        line_candidate3.append(669 - i)
                    '''
                elif (ECG[i][j] != 0) and (ECG[i-1][j] == 0) and (line == 2):
                    line = 0
                    break


            x = j - 165 - 300
            #print(x)
            #print(line_candidate1)
            #print(line_candidate2)
            #print(line_candidate3)

            tem_counts1 = np.bincount(line1)
            tem_counts2 = np.bincount(line2)
            tem_counts3 = np.bincount(line3)
                #返回众数
            tem_counts1 = np.argmax(tem_counts1)
            tem_counts2 = np.argmax(tem_counts2)
            tem_counts3 = np.argmax(tem_counts3)

            if line_candidate2 == []:
                line_candidate2 = line_candidate1
                line_candidate3 = line_candidate1

            elif line_candidate3 == []:
                line_candidate3 = line_candidate2

            
            if ((x == 0) or (x == 1)) :

                if (line_extreme_up == 1 and before_count_up == 0): #and (begin == 0) :
                    line1[x] = np.max(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)
                    before_count_up = 1
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                    #begin = 1
                    
                    
                elif (line_extreme_up == 0 and before_count_up == 1): #and (begin == 1):
                    line1[x] = 669 - 181
                    line2[x] = np.min(line_candidate1)
                    line3[x] = np.min(line_candidate2)
                    before_count_up = 1
                    before_count_down = 0
                    line_extreme_up = 0
                    line_extreme_down = 0


                elif (line_extreme_up == 1 and before_count_up == 1): #and (begin == 1) :
                    line1[x] = np.max(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)
                    before_count_up = 0
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                    #begin = 0


                elif (line_extreme_up == 0 and before_count_up == 0):
                    line1[x] = np.min(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)


            else:
                if (line_extreme_up == 1 and before_count_up == 0):
                    line1[x] = 669 - 181
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate2)
                        else :
                            line2[x] = np.min(line_candidate2)                
                    
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate3)
                    elif (line3[x-2]-line3[x-1] > 0):   
                        line3[x] = np.min(line_candidate3)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate3)
                        else :
                            line3[x] = np.min(line_candidate3)
                    
                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)

                        
                        #if line1[x] == (669 - 181 - 25) or 
                    if (line2[x-1] >= line2[x] + 120):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate1)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate1)
                            else :
                                line2[x] = np.min(line_candidate1)


                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)


                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)

                    before_count_up = 1
                    #before_count_down = 1
                    line_extreme_up = 0
                    #line_extreme_down = 0
                
                elif (line_extreme_up == 0 and before_count_up == 1):
                    line1[x] = 669 - 181
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate1)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate1)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate1)
                        else :
                            line2[x] = np.min(line_candidate1)

                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate2)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = np.min(line_candidate2)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate2)
                        else :
                            line3[x] = np.min(line_candidate2)

                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate1)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate1)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate1)
                            else :
                                line3[x] = np.min(line_candidate1)


                    '''
                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)
                    '''


                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)

                    before_count_up = 1
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                
                elif (line_extreme_up == 1 and before_count_up == 1):
                    line1[x] = 669 - 181


                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate2)
                        else :
                            line2[x] = np.min(line_candidate2)

                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate3)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = np.min(line_candidate3)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate3)
                        else :
                            line3[x] = np.min(line_candidate3)


                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)

                        
                        #if line1[x] == (669 - 181 - 25) or 
                    if (line2[x-1] >= line2[x] + 120):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate1)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate1)
                            else :
                                line2[x] = np.min(line_candidate1)
                    
                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)


                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)
                    line_extreme_up = 0
                    #line_extreme_down = 0
                    before_count_up = 0
                    #before_count_down = 1



                elif (line_extreme_up == 0 and before_count_up == 0):
                        
                    if (line1[x-1] - line2[x-1] >=0 ) and (line1[x-1] - line2[x-1] < (len(line_candidate1) + 7)) and ((np.min(line_candidate1) - np.max(line_candidate2)) >= 50) : 
                        line1[x] = np.min(line_candidate1)
                        line2[x] = np.max(line_candidate1)
                        #print(line_candidate1)
                        #print(line_candidate2)
                        #print(line_candidate3)
                        #print(line1[x-1] - line2[x-1])
                        #print(len(line_candidate1))
                        
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-3]-line3[x-2] > 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)   
                            
                            #if (line2[x] == np.max(line_candidate1)) or  (line2[x] == np.min(line_candidate1)) or 
                        if (line3[x-1] >= line3[x] + 120 ):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate1)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate1)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.max(line_candidate1)
                                elif (line3[x-3]-line3[x-2] > 0):
                                    line3[x] = np.min(line_candidate1)
                                else :
                                    line3[x] = np.min(line_candidate1)


                        if (line3[x] == line2[x]):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate2)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate2)
                                else :
                                    line3[x] = np.min(line_candidate2)
                                


                    elif (line2[x-1] - line3[x-1] >=0 ) and (line2[x-1] - line3[x-1] < (len(line_candidate2) + 7)) and ((np.min(line_candidate2) - np.max(line_candidate3)) >= 50) : 
                        #print(line2[x-1] - line3[x-1])
                        #print(len(line_candidate1))
                        if (line1[x-2]-line1[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line1[x] = np.max(line_candidate1)    
                        elif (line1[x-2]-line1[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line1[x] = np.min(line_candidate1)
                        elif (line1[x-2]-line1[x-1] == 0):
                            if (line1[x-3]-line1[x-2] < 0):
                                line1[x] = np.max(line_candidate1)
                            elif (line1[x-3]-line1[x-2] > 0):
                                line1[x] = np.min(line_candidate1)
                            else :
                                line1[x] = np.min(line_candidate1)
                        
                        line2[x] = np.min(line_candidate2)
                        line3[x] = np.max(line_candidate2)              
                    
                    else:
                        if (line1[x-2]-line1[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line1[x] = np.max(line_candidate1)
                            
                        elif (line1[x-2]-line1[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line1[x] = np.min(line_candidate1)
                        elif (line1[x-2]-line1[x-1] == 0):
                            if (line1[x-3]-line1[x-2] < 0):
                                line1[x] = np.max(line_candidate1)
                            elif (line1[x-3]-line1[x-2] > 0):
                                line1[x] = np.min(line_candidate1)
                            else :
                                line1[x] = np.min(line_candidate1)


                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.max(line_candidate2)
                            elif (line2[x-3]-line2[x-2] > 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)

                        
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.max(line_candidate3)
                            elif (line3[x-3]-line3[x-2] > 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)

                        #if (line2[x] == np.max(line_candidate1)) or  (line2[x] == np.min(line_candidate1)) or 
                        if (line3[x-1] >= line3[x] + 120 ):

                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate2)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate2)
                                else :
                                    line3[x] = np.min(line_candidate2)

                        
                        #if (line1[x] == 669 - 181) or 
                        if (line2[x-1] >= line2[x] + 120 ):
                            if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line2[x] = np.max(line_candidate1)
                            elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line2[x] = np.min(line_candidate1)
                            elif (line2[x-2]-line2[x-1] == 0):
                                if (line2[x-3]-line2[x-2] < 0):
                                    line2[x] = np.min(line_candidate1)
                                else :
                                    line2[x] = np.min(line_candidate1)

                        if (line2[x] == line1[x]):
                            if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line2[x] = np.max(line_candidate2)
                            elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line2[x] = np.min(line_candidate2)
                            elif (line2[x-2]-line2[x-1] == 0):
                                if (line2[x-3]-line2[x-2] < 0):
                                    line2[x] = np.min(line_candidate2)
                                else :
                                    line2[x] = np.min(line_candidate2)


                        if (line3[x] == line2[x]):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate3)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate3)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate3)
                                else :
                                    line3[x] = np.min(line_candidate3)


                    before_count_up = 0
                    before_count_down = 0
                    line_extreme_up = 0
                    line_extreme_down = 0
                
                
            line = 0
            tem_counts1 = 0
            tem_counts2 = 0
            tem_counts3 = 0
            line_candidate1 = []
            line_candidate2 = []
            line_candidate3 = []     

        high[3] = line1
        high[4] = line2
        high[5] = line3    
        #sio.savemat("line4.mat",{"line4":line1})
        #sio.savemat("line5.mat",{"line5":line2})
        #sio.savemat("line6.mat",{"line6":line3})

        ##第三列##

        #ECG = im
        line = 0
        line_extreme_up = 0
        #line_extreme_down = 0
        before_count_up = 0
        no_area = 0
        #before_count_down = 0
        line1_extreme = []
        line3_extreme = []
        line_candidate1 = []
        line_candidate2 = []
        line_candidate3 = []
        line1 = np.zeros(shape = 240,dtype=int)
        line2 = np.zeros(shape = 240,dtype=int)
        line3 = np.zeros(shape = 240,dtype=int)
        tem_counts1 = 0
        tem_counts2 = 0
        tem_counts3 = 0
        
        
        
        
        
        for j in range(165 + 300 * 2  ,165 + 300 * 2 + 240):
            for i  in range (181,670):
                if ECG[i][j] == 0 and line == 0:
                    #print(str(j)+ " " + str(i))
                    line_candidate1.append(669 - i)
                    if i == 181:
                        if before_count_up == 0:
                            line_extreme_up = 1
                            line1_extreme.append(j)
                        else :
                            line_extreme_up = 1
                            before_count_up = 1

                    '''
                    if len(line_candidate1) < 1:
                        line_candidate1.append(635 - i) 
                    elif (line_candidate1[len(line_candidate1)-1] - (635 - i)) == 1:
                        line_candidate1.append(635 - i)
                    '''
                elif (i >= 182 ) and(ECG[i][j] != 0) and (ECG[i-1][j] == 0) and (line == 0) :
                    line = line + 1
                    continue
                if ECG[i][j] == 0 and line == 1:
                    line_candidate2.append(669 - i)
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
                    line_candidate3.append(669 - i)
                    '''
                    if len(line_candidate3) < 1:
                        line_candidate3.append(669 - i) 
                    elif (line_candidate3[len(line_candidate3)-1] - (669 - i)) == 1:
                        line_candidate3.append(669 - i)
                    '''
                elif (ECG[i][j] != 0) and (ECG[i-1][j] == 0) and (line == 2):
                    line = 0
                    break


            x = j - 165 - 300 * 2
            #print(x)
            #print(line_candidate1)
            #print(line_candidate2)
            #print(line_candidate3)

            tem_counts1 = np.bincount(line1)
            tem_counts2 = np.bincount(line2)
            tem_counts3 = np.bincount(line3)
                #返回众数
            tem_counts1 = np.argmax(tem_counts1)
            tem_counts2 = np.argmax(tem_counts2)
            tem_counts3 = np.argmax(tem_counts3)
            

            if line_candidate2 == []:
                line_candidate2 = line_candidate1
                line_candidate3 = line_candidate1

            elif line_candidate3 == []:
                line_candidate3 = line_candidate2


            if ((x == 0) or (x == 1)) :

                if (line_extreme_up == 1 and before_count_up == 0): #and (begin == 0) :
                    line1[x] = np.max(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)
                    before_count_up = 1
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                    #begin = 1
                    
                    
                elif (line_extreme_up == 0 and before_count_up == 1): #and (begin == 1):
                    line1[x] = 669 - 181
                    line2[x] = np.min(line_candidate1)
                    line3[x] = np.min(line_candidate2)
                    before_count_up = 1
                    before_count_down = 0
                    line_extreme_up = 0
                    line_extreme_down = 0


                elif (line_extreme_up == 1 and before_count_up == 1): #and (begin == 1) :
                    line1[x] = np.max(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)
                    before_count_up = 0
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                    #begin = 0


                elif (line_extreme_up == 0 and before_count_up == 0):
                    line1[x] = np.min(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)


            else:
                if (line_extreme_up == 1 and before_count_up == 0):
                    line1[x] = 669 - 181
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate2)
                        else :
                            line2[x] = np.min(line_candidate2)                
                    
                    
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate3)
                    elif (line3[x-2]-line3[x-1] > 0):   
                        line3[x] = np.min(line_candidate3)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate3)
                        else :
                            line3[x] = np.min(line_candidate3)


                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)

                        
                        #if line1[x] == (669 - 181 - 25) or 
                    if (line2[x-1] >= line2[x] + 120):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate1)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate1)
                            else :
                                line2[x] = np.min(line_candidate1)


                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)


                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)


                    before_count_up = 1
                    #before_count_down = 1
                    line_extreme_up = 0
                    #line_extreme_down = 0
                
                elif (line_extreme_up == 0 and before_count_up == 1):
                    line1[x] = 669 - 181
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate1)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate1)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate1)
                        else :
                            line2[x] = np.min(line_candidate1)

                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate2)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = np.min(line_candidate2)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate2)
                        else :
                            line3[x] = np.min(line_candidate2)


                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate1)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate1)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate1)
                            else :
                                line3[x] = np.min(line_candidate1)

                    '''
                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)
                    '''

                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)

                    before_count_up = 1
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                
                elif (line_extreme_up == 1 and before_count_up == 1):
                    line1[x] = 669 - 181


                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate2)
                        else :
                            line2[x] = np.min(line_candidate2)

                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate3)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = np.min(line_candidate3)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate3)
                        else :
                            line3[x] = np.min(line_candidate3)


                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)

                        
                        #if line1[x] == (669 - 181 - 25) or 
                    if (line2[x-1] >= line2[x] + 120):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate1)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate1)
                            else :
                                line2[x] = np.min(line_candidate1)


                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)


                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)
                    line_extreme_up = 0
                    #line_extreme_down = 0
                    before_count_up = 0
                    #before_count_down = 1



                elif (line_extreme_up == 0 and before_count_up == 0):

                    if (line1[x-1] - line2[x-1] >=0 ) and (line1[x-1] - line2[x-1] < (len(line_candidate1) + 7)) and ((np.min(line_candidate1) - np.max(line_candidate2)) >= 50) : 
                        line1[x] = np.min(line_candidate1)
                        line2[x] = np.max(line_candidate1)
                        #print(line1[x-1] - line2[x-1])
                        #print(len(line_candidate1))
                        
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-3]-line3[x-2] > 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)  

                        if (line3[x-1] >= line3[x] + 120):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate1)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate1)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] > 0):
                                    line3[x] = np.min(line_candidate1)
                                else :
                                    line3[x] = np.min(line_candidate1)

                        if (line3[x] == line2[x]):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate2)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate2)
                                else :
                                    line3[x] = np.min(line_candidate2)

                    elif (line2[x-1] - line3[x-1] >=0 ) and (line2[x-1] - line3[x-1] < (len(line_candidate2) + 7)): #and ((np.min(line_candidate2) - np.max(line_candidate3)) >= 50) : 
                        
                        #print(line2[x-1] - line3[x-1])
                        #print(len(line_candidate1))
                        if (line1[x-2]-line1[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line1[x] = np.max(line_candidate1)    
                        elif (line1[x-2]-line1[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line1[x] = np.min(line_candidate1)
                        elif (line1[x-2]-line1[x-1] == 0):
                            if (line1[x-3]-line1[x-2] < 0):
                                line1[x] = np.max(line_candidate1)
                            elif (line1[x-3]-line1[x-2] > 0):
                                line1[x] = np.min(line_candidate1)
                            else :
                                line1[x] = np.min(line_candidate1)
                        
                        line2[x] = np.min(line_candidate2)
                        line3[x] = np.max(line_candidate2)              
                    
                    else:
                        if (line1[x-2]-line1[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line1[x] = np.max(line_candidate1)
                            
                        elif (line1[x-2]-line1[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line1[x] = np.min(line_candidate1)
                        elif (line1[x-2]-line1[x-1] == 0):
                            if (line1[x-3]-line1[x-2] < 0):
                                line1[x] = np.max(line_candidate1)
                            elif (line1[x-3]-line1[x-2] > 0):
                                line1[x] = np.min(line_candidate1)
                            else :
                                line1[x] = np.min(line_candidate1)


                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.max(line_candidate2)
                            elif (line2[x-3]-line2[x-2] > 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)

                        
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.max(line_candidate3)
                            elif (line3[x-3]-line3[x-2] > 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)
                        
                        #if (line2[x] == np.max(line_candidate1)) or  (line2[x] == np.min(line_candidate1)) or 
                        if (line3[x-1] >= line3[x] + 120):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate2)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] > 0):
                                    line3[x] = np.min(line_candidate2)
                                else :
                                    line3[x] = np.min(line_candidate2)

                        #if line1[x] == (669 - 181 - 25) or 
                        if (line2[x-1] >= line2[x] + 120):
                            if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line2[x] = np.max(line_candidate1)
                            elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line2[x] = np.min(line_candidate1)
                            elif (line2[x-2]-line2[x-1] == 0):
                                if (line2[x-3]-line2[x-2] < 0):
                                    line2[x] = np.min(line_candidate1)
                                else :
                                    line2[x] = np.min(line_candidate1)


                        if (line2[x] == line1[x]):
                            if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line2[x] = np.max(line_candidate2)
                            elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line2[x] = np.min(line_candidate2)
                            elif (line2[x-2]-line2[x-1] == 0):
                                if (line2[x-3]-line2[x-2] < 0):
                                    line2[x] = np.min(line_candidate2)
                                else :
                                    line2[x] = np.min(line_candidate2)


                        if (line3[x] == line2[x]):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate3)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate3)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate3)
                                else :
                                    line3[x] = np.min(line_candidate3)


                    before_count_up = 0
                    before_count_down = 0
                    line_extreme_up = 0
                    line_extreme_down = 0
                
                
            line = 0
            tem_counts1 = 0
            tem_counts2 = 0
            tem_counts3 = 0
            line_candidate1 = []
            line_candidate2 = []
            line_candidate3 = []     
        #sio.savemat("line7.mat",{"line7":line1})
        #sio.savemat("line8.mat",{"line8":line2})
        #sio.savemat("line9.mat",{"line9":line3})
        high[6] = line1
        high[7] = line2
        high[8] = line3

        ##第四列##

        #ECG = im
        line = 0
        line_extreme_up = 0
        #line_extreme_down = 0
        before_count_up = 0
        no_area = 0
        #before_count_down = 0
        line1_extreme = []
        line3_extreme = []
        line_candidate1 = []
        line_candidate2 = []
        line_candidate3 = []
        line1 = np.zeros(shape = 240,dtype=int)
        line2 = np.zeros(shape = 240,dtype=int)
        line3 = np.zeros(shape = 240,dtype=int)
        tem_counts1 = 0
        tem_counts2 = 0
        tem_counts3 = 0
        for j in range(165 + 300 * 3  ,165 + 300 * 3 + 240):
            for i  in range (181 + 25 ,670):
                if ECG[i][j] == 0 and line == 0:
                    #print(str(j)+ " " + str(i))
                    line_candidate1.append(669 - i)
                    if i == 181 + 25:
                        if before_count_up == 0:
                            line_extreme_up = 1
                            line1_extreme.append(j)
                        else :
                            line_extreme_up = 1
                            before_count_up = 1
                    
                    '''
                    if len(line_candidate1) < 1:
                        line_candidate1.append(635 - i) 
                    elif (line_candidate1[len(line_candidate1)-1] - (635 - i)) == 1:
                        line_candidate1.append(635 - i)
                    '''
                elif (i >= 182 + 25 ) and (ECG[i][j] != 0) and (ECG[i-1][j] == 0) and (line == 0) :
                    line = line + 1
                    continue
                if ECG[i][j] == 0 and line == 1:
                    line_candidate2.append(669 - i)
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
                    line_candidate3.append(669 - i)
                    '''
                    if len(line_candidate3) < 1:
                        line_candidate3.append(669 - i) 
                    elif (line_candidate3[len(line_candidate3)-1] - (669 - i)) == 1:
                        line_candidate3.append(669 - i)
                    '''
                elif (ECG[i][j] != 0) and (ECG[i-1][j] == 0) and (line == 2):
                    line = 0
                    break


            x = j - 165 - 300 * 3
            #print(x)
            #print(line_candidate1)
            #print(line_candidate2)
            #print(line_candidate3)

            tem_counts1 = np.bincount(line1)
            tem_counts2 = np.bincount(line2)
            tem_counts3 = np.bincount(line3)
                #返回众数
            tem_counts1 = np.argmax(tem_counts1)
            tem_counts2 = np.argmax(tem_counts2)
            tem_counts3 = np.argmax(tem_counts3)
        
            
            if line_candidate2 == []:
                line_candidate2 = line_candidate1
                line_candidate3 = line_candidate1

            elif line_candidate3 == []:
                line_candidate3 = line_candidate2


            if ((x == 0) or (x == 1)) :

                if (line_extreme_up == 1 and before_count_up == 0): #and (begin == 0) :
                    line1[x] = np.max(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)
                    before_count_up = 1
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                    #begin = 1
                    
                    
                elif (line_extreme_up == 0 and before_count_up == 1): #and (begin == 1):
                    line1[x] = 669 - 181 - 25
                    line2[x] = np.min(line_candidate1)
                    line3[x] = np.min(line_candidate2)
                    before_count_up = 1
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0


                elif (line_extreme_up == 1 and before_count_up == 1): #and (begin == 1) :
                    line1[x] = np.max(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)
                    before_count_up = 0
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                    #begin = 0


                elif (line_extreme_up == 0 and before_count_up == 0):
                    line1[x] = np.min(line_candidate1)
                    line2[x] = np.min(line_candidate2)
                    line3[x] = np.min(line_candidate3)


            else :

                if (line_extreme_up == 1 and before_count_up == 0):
                    line1[x] = 669 - 181 - 25
                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate2)
                        else :
                            line2[x] = np.min(line_candidate2)                
                    
                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate3)
                    elif (line3[x-2]-line3[x-1] > 0):   
                        line3[x] = np.min(line_candidate3)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate3)
                        else :
                            line3[x] = np.min(line_candidate3)


                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)

                        
                        #if line1[x] == (669 - 181 - 25) or 
                    if (line2[x-1] >= line2[x] + 120):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate1)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate1)
                            else :
                                line2[x] = np.min(line_candidate1)

                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)


                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)


                    before_count_up = 1
                    #before_count_down = 1
                    line_extreme_up = 0
                    #line_extreme_down = 0
                
                elif (line_extreme_up == 0 and before_count_up == 1):
                    #print(j,i)
                    #print(line_candidate1)
                    #print(line_candidate2)
                    #print(line_candidate3)
                    line1[x] = 669 - 181 -25

                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate1)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate1)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate1)
                        else :
                            line2[x] = np.min(line_candidate1)

                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate2)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = np.min(line_candidate2)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate2)
                        else :
                            line3[x] = np.min(line_candidate2)
                    
                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate1)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate1)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate1)
                            else :
                                line3[x] = np.min(line_candidate1)

                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)

                    before_count_up = 1
                    #before_count_down = 0
                    line_extreme_up = 0
                    #line_extreme_down = 0
                
                elif (line_extreme_up == 1 and before_count_up == 1):
                    line1[x] = 669 - 181 - 25


                    if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line2[x] = np.max(line_candidate2)
                    elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line2[x] = np.min(line_candidate2)
                    elif (line2[x-2]-line2[x-1] == 0):
                        if (line2[x-3]-line2[x-2] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-3]-line2[x-2] > 0):
                            line2[x] = np.min(line_candidate2)
                        else :
                            line2[x] = np.min(line_candidate2)

                    if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                        line3[x] = np.max(line_candidate3)
                    elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                        line3[x] = np.min(line_candidate3)
                    elif (line3[x-2]-line3[x-1] == 0):
                        if (line3[x-3]-line3[x-2] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-3]-line3[x-2] > 0):
                            line3[x] = np.min(line_candidate3)
                        else :
                            line3[x] = np.min(line_candidate3)


                    if (line3[x-1] >= line3[x] + 120):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)

                        
                        #if line1[x] == (669 - 181 - 25) or 
                    if (line2[x-1] >= line2[x] + 120):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate1)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate1)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate1)
                            else :
                                line2[x] = np.min(line_candidate1)

                    if (line2[x] == line1[x]):
                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)


                    if (line3[x] == line2[x]):
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)


                    line_extreme_up = 0
                    #line_extreme_down = 0
                    before_count_up = 0
                    #before_count_down = 1



                elif (line_extreme_up == 0 and before_count_up == 0):
                    if (line1[x-1] - line2[x-1] >=0 ) and (line1[x-1] - line2[x-1] < (len(line_candidate1) + 7)) and ((np.min(line_candidate1) - np.max(line_candidate2)) >= 50) : 
                        line1[x] = np.min(line_candidate1)
                        line2[x] = np.max(line_candidate1)
                        #print(line1[x-1] - line2[x-1])
                        #print(len(line_candidate1))
                        
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate2)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate2)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-3]-line3[x-2] > 0):
                                line3[x] = np.min(line_candidate2)
                            else :
                                line3[x] = np.min(line_candidate2)   
                            
                            #if (line2[x] == np.max(line_candidate1)) or  (line2[x] == np.min(line_candidate1)) or 
                        if (line3[x-1] >= line3[x] + 120):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate1)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate1)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate1)
                                else :
                                    line3[x] = np.min(line_candidate1)

                        if (line3[x] == line2[x]):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate2)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate2)
                                else :
                                    line3[x] = np.min(line_candidate2)


                    elif (line2[x-1] - line3[x-1] >=0 ) and (line2[x-1] - line3[x-1] < (len(line_candidate2) + 7)) and ((np.min(line_candidate2) - np.max(line_candidate3)) >= 50) : 
                        #print(line2[x-1] - line3[x-1])
                        #print(len(line_candidate1))
                        if (line1[x-2]-line1[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line1[x] = np.max(line_candidate1)    
                        elif (line1[x-2]-line1[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line1[x] = np.min(line_candidate1)
                        elif (line1[x-2]-line1[x-1] == 0):
                            if (line1[x-3]-line1[x-2] < 0):
                                line1[x] = np.max(line_candidate1)
                            elif (line1[x-3]-line1[x-2] > 0):
                                line1[x] = np.min(line_candidate1)
                            else :
                                line1[x] = np.min(line_candidate1)
                        
                        line2[x] = np.min(line_candidate2)
                        line3[x] = np.max(line_candidate2)              
                    
                    else:
                        if (line1[x-2]-line1[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line1[x] = np.max(line_candidate1)
                            
                        elif (line1[x-2]-line1[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line1[x] = np.min(line_candidate1)
                        elif (line1[x-2]-line1[x-1] == 0):
                            if (line1[x-3]-line1[x-2] < 0):
                                line1[x] = np.max(line_candidate1)
                            elif (line1[x-3]-line1[x-2] > 0):
                                line1[x] = np.min(line_candidate1)
                            else :
                                line1[x] = np.min(line_candidate1)


                        if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line2[x] = np.max(line_candidate2)
                        elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line2[x] = np.min(line_candidate2)
                        elif (line2[x-2]-line2[x-1] == 0):
                            if (line2[x-3]-line2[x-2] < 0):
                                line2[x] = np.max(line_candidate2)
                            elif (line2[x-3]-line2[x-2] > 0):
                                line2[x] = np.min(line_candidate2)
                            else :
                                line2[x] = np.min(line_candidate2)

                        
                        if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                            line3[x] = np.max(line_candidate3)
                        elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                            line3[x] = np.min(line_candidate3)
                        elif (line3[x-2]-line3[x-1] == 0):
                            if (line3[x-3]-line3[x-2] < 0):
                                line3[x] = np.max(line_candidate3)
                            elif (line3[x-3]-line3[x-2] > 0):
                                line3[x] = np.min(line_candidate3)
                            else :
                                line3[x] = np.min(line_candidate3)
                    

                        #if (line2[x] == np.max(line_candidate1)) or  (line2[x] == np.min(line_candidate1)) or 
                        if (line3[x-1] >= line3[x] + 120):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate2)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate2)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate2)
                                else :
                                    line3[x] = np.min(line_candidate2)

                        
                        #if line1[x] == (669 - 181 - 25) or 
                        if (line2[x-1] >= line2[x] + 120):
                            if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line2[x] = np.max(line_candidate1)
                            elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line2[x] = np.min(line_candidate1)
                            elif (line2[x-2]-line2[x-1] == 0):
                                if (line2[x-3]-line2[x-2] < 0):
                                    line2[x] = np.min(line_candidate1)
                                else :
                                    line2[x] = np.min(line_candidate1)
                        
                        if (line2[x] == line1[x]):
                            if (line2[x-2]-line2[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line2[x] = np.max(line_candidate2)
                            elif (line2[x-2]-line2[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line2[x] = np.min(line_candidate2)
                            elif (line2[x-2]-line2[x-1] == 0):
                                if (line2[x-3]-line2[x-2] < 0):
                                    line2[x] = np.min(line_candidate2)
                                else :
                                    line2[x] = np.min(line_candidate2)


                        if (line3[x] == line2[x]):
                            if (line3[x-2]-line3[x-1] < 0):# or (High[x-3]-High[x-1] < 0):
                                line3[x] = np.max(line_candidate3)
                            elif (line3[x-2]-line3[x-1] > 0):# or (High[x-3]-High[x-1] > 0):
                                line3[x] = np.min(line_candidate3)
                            elif (line3[x-2]-line3[x-1] == 0):
                                if (line3[x-3]-line3[x-2] < 0):
                                    line3[x] = np.min(line_candidate3)
                                else :
                                    line3[x] = np.min(line_candidate3)
                
                
            line = 0
            tem_counts1 = 0
            tem_counts2 = 0
            tem_counts3 = 0
            line_candidate1 = []
            line_candidate2 = []
            line_candidate3 = []     
        #sio.savemat("line10.mat",{"line10":line1})
        #sio.savemat("line11.mat",{"line11":line2})
        #sio.savemat("line12.mat",{"line12":line3})
        high[9] = line1
        high[10] = line2
        high[11] = line3


    ######################################################################################


    T1 = np.zeros(shape = 6)
    T2 = np.zeros(shape = 6)
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


    for k in range(12):
        Q_final = 0
        S_final = 0
        A = 0
        A1 = 0
        A2 = 0
        B = 0
        C = 0

        High = np.zeros(shape =240)
        #print('Picture ' + name[k] + ':')
        #sio.savemat("line"+str(k)+".mat",{"line"+str(k):High})
        High = high[k]
        
        for i in range(2,237):
            if (abs(High[i+1]-High[i]) + abs(High[i-1]-High[i])) >= 100:
                High[i] = (High[i+1] + High[i-1])/2
                #print("good")
        sio.savemat("High"+str(k)+".mat",{"High":High})
        A1 = np.max(High)             
        A2 = np.min(High)
        counts = np.bincount(High)
            #返回众数
        A = np.argmax(counts)
        #print("Up or down: " + str(B)+"," + str(C))

        #G =  - A2
        #M = A1 - base
        R_index1 = 0
        R_index2 = 0
        #print(High)
        down = 0
        #print(base)
        if (A - A2) > (A1 - A):
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
                
                
            #print(np.where(High==1))
            #print(High_extreme_down)
            #print(C)
                    
            

            
            
            R_index = int(np.where(High==np.min(High))[0][0])
            
            while R_index <= 50 or R_index >=200:
                
                High[R_index] = A
                
                New_index = int(np.where(High==np.min(High))[0][0])
                #print("New_index"+ str(New_index))
                R_index = New_index
                #print(R_index)
            R_index1 = R_index - 2
            R_index2 = R_index + 2
            if np.max(High)>=668:
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


            Base = np.bincount(High[R_index1-40:R_index2+20])###!!!!!change
            #print(Base)
            if len(Base) == 0:
                Base = np.bincount(High)###!!!!!change
            base = np.argmax(Base)
            #print(base)
            #print(R_index1)
            #print(R_index2)

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
            #    S = int(np.min(S_index))
            #else:
            #    S = int(np.median(S_index))

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
            #    Q = int(np.max(Q_index))
            #else:
            #    Q = int(np.median(Q_index))

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
                        Qs = i + 3
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
                        Qs = i + 3
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
                            Qs = i + 3
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
                            Qs = i + 3
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
                            Qs = i + 3
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
                            Qs = i + 3
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
                            Qs = i + 3
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
                            Qs = i + 3
                            break

            if Qs == 0:
                Qs = Q_final - 3
            
            num1 = 0
            num2 = 0
            up = 0
                  

            #print("Qs: "+ str(Qs))
            #Qs = int(np.median(Qs))



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
            #print(np.where(High==119))
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
            #    R_index1 = int(np.where(High==np.max(High))[0][2]) #High_extreme[2]
            #    R_index2 = int(np.where(High==np.max(High))[0][3]) #High_extreme[3]
                #print(np.where(High==1))
        

            R_index = int(np.where(High==np.max(High))[0][0]) 
            #print(np.where(High==np.max(High)))
            while R_index <= 50 or R_index >=200:
                High[R_index] = A
                New_index = int(np.where(High==np.max(High))[0][0])
                #print(New_index)
                R_index = New_index
            R_index1 = R_index - 2
            R_index2 = R_index + 2
            if np.max(High)>=668:
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
            
            Base = np.bincount(High[R_index1-40:R_index2+20]) 
            #print(Base)
            if len(Base) == 0:
                Base = np.bincount(High)###!!!!!change  ###!!!!!change the range
            base = np.argmax(Base)
            #print(base)
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
            #    S = int(np.min(S_index))
            #else:
            #    S = int(np.median(S_index))

        
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
                        Qs = i + 3
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
                        Qs = i + 3
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
                            Qs = i + 3
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
                            Qs = i + 3
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
                            Qs = i + 3
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
                            Qs = i + 3
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
                            Qs = i + 3
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
                            Qs = i + 3
                            break

            if Qs == 0:
                Qs = Q_final - 3

            
                         
            num1 = 0
            num2 = 0
            up = 0
            #print(Qs)
            #Qs = int(np.median(Qs))

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
                I_max = np.max(High[Q_final:S_final]) - High[S_final]
            if k == 6:
                PvN = (np.max(High[Q_final:S_final]) - High[S_final]) / (High[S_final] - np.min(High[Q_final:S_final]))
                V1_max = np.max(High[Q_final:S_final]) - High[S_final]
                
            ###################################################
            if k == 4:
                aVL_max = np.max(High[Q_final:S_final])  - High[S_final]
                
            

            
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
            
            I_max = np.max(High[Q_final:S_final]) - High[S_final]
        
        if k == 6:
            '''
            print("Qt: "+str(Qt)+"; Q: "+str(Q_final)+"; S: "+str(S_final))
            print("Q_index: "+str(Q_index)+"; S_index: "+str(S_index))
            
            print("np.max(High[Q_final:S_final])"+str(np.max(High[Q_final:S_final])))
            print("High[S_final]"+ str(High[S_final]))
            print("##########"+str(np.max(High[Qt:S_final])))
            print("##########"+str(np.max(High[Qt:S_final])))
            '''
            V1_max = np.max(High[Qt:S_final]) - High[S_final]
            A = abs(np.max(High[Qt:S_final])- High[S_final])
            B = abs(High[S_final] - np.min(High[Qt:S_final]))
            if B == 0:
                B = 1
            PvN = A/B
        if k == 4:
            
            aVL_max = np.max(High[Q_final:S_final]) - High[S_final]
    
    result = np.zeros(shape = 25)
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
    

    #return #T1, T2, Positive1, Positive2, PvN, I_minus, II_minus, III_minus, abs(area), V1, V2, V3, V4, V5, V6
    return result#np.array([T1[0],T1[1],T1[2],T1[3],T1[4],T1[5],T2[0],T2[1],T2[2],T2[3],T2[4],T2[5],Positive1,Positive2,PvN,I_minus,II_minus,III_minus,area,V1,V2,V3,V4,V5,V6])

    '''
    print("T1_final: "+str(T1_final))
    print("T2_final: "+str(T2_final))

    print("Positive1: "+str(Positive1))
    print("Positive2: "+str(Positive2))

    print("PvN: "+str(PvN))
    I_minus = I_max - V1_max
    II_minus = aVL_max - V1_max
    III_minus = I_max + aVL_max - V1_max

    #print("Positive: "+str(Positive))
    print("I_max: "+str(I_max))
    print("aVL_max: "+str(aVL_max))
    print("V1_max: "+str(V1_max))
    print("I_minus: "+str(I_minus))
    print("II_minus: "+str(II_minus))
    print("III_minus: "+str(III_minus))
    print("II: "+str(II))
    print("III: "+str(III))

    print("V1: "+str(V1))
    print("V2: "+str(V2))
    print("V3: "+str(V3))
    print("V4: "+str(V4))
    print("V5: "+str(V5))
    print("V6: "+str(V6))
    '''   



if __name__ == "__main__":
    A = './A'#存放图片的文件夹路径
    pathsA = glob.glob(os.path.join(A, '*.jpg'))
    B = './B'
    pathsB = glob.glob(os.path.join(B, '*.jpg'))
    N = './N'
    pathsN = glob.glob(os.path.join(N, '*.jpg'))
    T = './T'
    pathsT = glob.glob(os.path.join(T, '*.jpg'))
    T_A = './T_A'
    pathsT_A = glob.glob(os.path.join(T_A, '*.jpg'))
    T_B = './T_B'
    pathsT_B = glob.glob(os.path.join(T_B, '*.jpg'))
    T_N = './T_N'
    pathsT_N = glob.glob(os.path.join(T_N, '*.jpg'))
    pathsA.sort()
    pathsB.sort()
    pathsN.sort()
    pathsT.sort()
    pathsT_A.sort()
    pathsT_B.sort()
    pathsT_N.sort()


    #result = np.zeros(shape = 25)
    A_len = len(pathsA)
    #label1 = np.zeros(shape = A_len)
    #data1 = np.zeros( shape = (A_len,25))
    with open("resultA.txt", "w") as f:
        for i, path in enumerate(pathsA):
            im = np.array(Image.open(path).convert('L'))
        
            output1 = feature(im)
            #for k in range(len(result)):
            if i == 0:
                data = output1
                label = 0 
            else:
                data = np.row_stack((data, output1))  
                label = np.append(label,0)
            print("A: " + str(i))
            #print(result)
            f.write('name: '+ str(path))
            f.write('\n')
            f.write('feature: ' + str(output1))
            f.flush()

    B_len = len(pathsB)
    #label2 = np.ones(shape = B_len)
    #data2 = np.zeros( shape = (B_len,25))
    with open("resultB.txt", "w") as f:
        for i, path in enumerate(pathsB):
            im = np.array(Image.open(path).convert('L'))
            output2 = feature(im)
            data = np.row_stack((data, output2))
            label = np.append(label,1)
            #for k in range(len(result)):
            #    data = np.append(data,result)
            #print(result)
            print("B: " + str(i))
            f.write('name: '+ str(path))
            f.write('\n')
            f.write('feature: ' + str(output2))
            f.flush()


    N_len = len(pathsN)

    with open("resultC.txt", "w") as f:
    #data3 = np.zeros( shape = (N_len,25))
        for i, path in enumerate(pathsN):
            im = np.array(Image.open(path).convert('L'))
       
            output3 = feature(im)
            data = np.row_stack((data, output3))
            label = np.append(label,2)
            
            #for k in range(len(result)):
            #    data3[i,k] = result[k]
            #print(result)
            print("N: " + str(i))
            f.write('name: '+ str(path))
            f.write('\n')
            f.write('feature: ' + str(output3))
            f.flush()
    '''
    label = np.append(label1,label2)
    label = np.append(label,label3)
    data = np.append(data1,data2)
    data = np.append(data,data3)
    '''
    print(len(data))
    print(len(label))
    model = SVC(kernel='poly',gamma='auto',probability=False)#probability=False时，没办法调用 model.predict_proba()函数
    model.fit(data,label)


    
    #T_len = len(pathsT)
    #labelT = np.zeros(shape = T_len)
    #dataT = np.zeros( shape = (T_len,25))
    with open("resultT.txt", "w") as f:
        for i, path in enumerate(pathsT):
            im = np.array(Image.open(path).convert('L'))
            outputT = feature(im)
            if i == 0:
                dataT = outputT
            else:
                dataT = np.row_stack((dataT, outputT)) 
            #for k in range(len(result)):
            #    dataT[i,k] = result[k]
            #print(result)
            print("T: " + str(i))
            f.write('name: '+ str(path))
            f.write('\n')
            f.write('feature: ' + str(outputT))
            f.flush()
    
    #print(len(dataT))
    #print(model.predict_proba(data))
    pre1 = model.predict(dataT)
    print("This is dataT result")
    print(pre1)


    with open("resultT.txt", "w") as f:
        for i, path in enumerate(pathsT_A):
            im = np.array(Image.open(path).convert('L'))
            outputT_A = feature(im)
            if i == 0:
                dataT_A = outputT_A
            else:
                dataT_A = np.row_stack((dataT_A, outputT_A)) 
            #for k in range(len(result)):
            #    dataT[i,k] = result[k]
            #print(result)
            print("T: " + str(i))
            f.write('name: '+ str(path))
            f.write('\n')
            f.write('feature: ' + str(outputT_A))
            f.flush()
    
    #print(len(dataT))
    #print(model.predict_proba(data))
    pre2 = model.predict(dataT_A)
    print("This is dataA result")
    print(pre2)


    with open("resultT.txt", "w") as f:
        for i, path in enumerate(pathsT_B):
            im = np.array(Image.open(path).convert('L'))
            outputT_B = feature(im)
            if i == 0:
                dataT_B = outputT_B
            else:
                dataT_B = np.row_stack((dataT_B, outputT_B)) 
            #for k in range(len(result)):
            #    dataT[i,k] = result[k]
            #print(result)
            print("T: " + str(i))
            f.write('name: '+ str(path))
            f.write('\n')
            f.write('feature: ' + str(outputT_B))
            f.flush()
    
    #print(len(dataT))
    #print(model.predict_proba(data))
    pre3 = model.predict(dataT_B)
    print("This is dataB result")
    print(pre3)

    with open("resultT.txt", "w") as f:
        for i, path in enumerate(pathsT_N):
            im = np.array(Image.open(path).convert('L'))
            outputT_N = feature(im)
            if i == 0:
                dataT_N = outputT_N
            else:
                dataT_N = np.row_stack((dataT_N, outputT_N)) 
            #for k in range(len(result)):
            #    dataT[i,k] = result[k]
            #print(result)
            print("T: " + str(i))
            f.write('name: '+ str(path))
            f.write('\n')
            f.write('feature: ' + str(outputT_N))
            f.flush()
    
    #print(len(dataT))
    #print(model.predict_proba(data))
    pre4 = model.predict(dataT_N)
    print("This is dataN result")
    print(pre4)



#data = np.array([[2,0],[0,2],[0,0],[3,0],[0,3],[3,3]])
#label = np.array([1,1,1,2,2,2])
#model = SVC(kernel='sigmoid', probability=True)#probability=False时，没办法调用 model.predict_proba()函数
#model.fit(data,label)
#C = [[-1,-1],[4,4]]
#pre = model.predict_proba(C)
#print(pre)
#pre1 = model.predict(C)
#print(pre1)
