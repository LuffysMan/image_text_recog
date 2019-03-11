import cv2
import math
import numpy as np
import os
import glob
from math import fabs, sin, cos, acos, radians

IMAGE_TRAIN_PATH = 'dataset/image_train'
TXT_TRAIN_PATH = 'dataset/txt_train'
IMAGE_TRAIN_PROD_PATH = 'dataset/img_train_prod'     #预处理后的图像路径
TXT_TRAIN_PROD_PATH = 'dataset/txt_train_prod'       #预处理后的图像对应文本路径

'''旋转图像并剪裁'''
def rotate(img,                    # 图片
           pt1, pt2, pt3, pt4,     # 四点坐标
           NewimageName):            # 输出图片路径
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)      # 矩形框的宽度
#    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
    if(withRect!=0):
        angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)               # 矩形框旋转角度
    
        if pt4[1]<pt1[1]:
            angle=-angle
        
        height = img.shape[0]  # 原始图像高度
        width = img.shape[1]   # 原始图像宽度
        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)     # 按angle角度旋转图像
        heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
    
        rotateMat[0, 2] += (widthNew - width) / 2
        rotateMat[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
    
        # 旋转后图像的四点坐标
        [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
        [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
        [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))
    
        # 处理反转的情况
        if pt2[1]>pt4[1]:
            pt2[1],pt4[1]=pt4[1],pt2[1]
        if pt1[0]>pt3[0]:
            pt1[0],pt3[0]=pt3[0],pt1[0]
    
        imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
        cv2.imwrite(NewimageName, imgOut)  # 保存得到的旋转后的矩形框
        return imgRotation                 # rotated image
 
#  读取四点坐标
def ReadTxt(directory,ext):
    global image_path_prod, txt_path_prod, allpic, nowimage, nowtxt, nowline, invalidimg
    os.chdir(os.path.join(os.getcwd(), IMAGE_TRAIN_PATH))   #修改当前工作路径, 方便获取文件名
    image_names_train = glob.glob('*.jpg')                     #获取工作路径下所有jpg格式文件名到list中
    count_img = len(image_names_train) 
    print("image count: {}".format(count_img))
    for j in range(count_img):
        print('处理图片:'+str(j))
        imageTxt = os.path.join(directory, image_names_train[j][:-4] + ext)     # txt路径
        imageName =image_names_train[j]
        nowimage = imageName
        nowtxt = imageTxt
        nowline = 0
        imgSrc = cv2.imread(imageName)
        if(imgSrc is None):
            invalidimg.append(nowimage)
        else:
            F = open(imageTxt,'rb')								#以二进制模式打开目标txt文件
            lines = F.readlines()								#逐行读入内容
            length=len(lines)
            s = 0                                               #计算图片编号，对应文本描述
            for i in range(length):
                lines[i] = str(lines[i], encoding = "utf-8")    #从bytes转为str格式
                des = lines[i].split(',')[-1:]
                nowline = i
                if ((des != ['###\n']) and (des != ['###'])):
                    s = s + 1
                    allpic+=1
                    #保存新图片/txt格式为"原名字+编号+.jpg/.txt"
                    NewimageName = os.path.join(image_path_prod, imageName[:-3] + str(s) + '.jpg')
                    NewtxtName = os.path.join(txt_path_prod, imageName[:-3] + str(s) + '.txt')
                    #写入新TXT文件
                    if (s == length):
                        des = str(des)[2:-2]
                    else:
                        des = str(des)[2:-4]
                    file = open(NewtxtName,'w')				#打开or创建一个新的txt文件
                    file.write(des)        					#写入内容信息  
                    file.close()  
                    # str转float
                    pt1 = list(map(float,lines[i].split(',')[:2]))
                    pt2 = list(map(float,lines[i].split(',')[2:4]))
                    pt3 = list(map(float,lines[i].split(',')[4:6]))
                    pt4 = list(map(float,lines[i].split(',')[6:8]))
                    # float转int 
                    pt1=list(map(int,pt1))
                    pt2=list(map(int,pt2))        
                    pt4=list(map(int,pt4))
                    pt3=list(map(int,pt3))
                    rotate(imgSrc,pt1,pt2,pt3,pt4,NewimageName)                
    
if __name__=="__main__":
    cur_path = os.getcwd()
    print("cur_path: {}".format(cur_path))
    image_path_prod = os.path.join(cur_path, "dataset/img_train_prod")
    txt_path_prod = os.path.join(cur_path, "dataset/txt_train_prod")
    if not os.path.exists(image_path_prod):
        os.mkdir(image_path_prod)
    if not os.path.exists(txt_path_prod):
        os.mkdir(txt_path_prod)
    allpic = 0
    nowimage = ''
    nowtxt = ''
    nowline = 0
    invalidimg=[]
    directory = os.path.join(cur_path, TXT_TRAIN_PATH) #TXT文件路径
    ext = '.txt'
    ReadTxt(directory,ext)