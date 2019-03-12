import cv2
import math
import numpy as np
import os
import glob
from math import fabs, sin, cos, acos, radians

import threading
import queue
import multiprocessing

IMAGE_TRAIN_PATH = 'dataset/image_train'
TXT_TRAIN_PATH = 'dataset/txt_train'
IMAGE_TRAIN_PROD_PATH = 'dataset/img_train_prod'     #预处理后的图像路径
TXT_TRAIN_PROD_PATH = 'dataset/txt_train_prod'       #预处理后的图像对应文本路径
g_img_train_path = os.path.join(os.getcwd(), IMAGE_TRAIN_PATH)
g_txt_train_path = os.path.join(os.getcwd(), TXT_TRAIN_PATH)
g_img_train_prod_path = os.path.join(os.getcwd(), IMAGE_TRAIN_PROD_PATH)
g_txt_train_prod_path = os.path.join(os.getcwd(), TXT_TRAIN_PROD_PATH)

g_exitFlag = 0

class myThread(threading.Thread):
    __threadCount = 0
    __lock = threading.Lock()
    def __init__(self, name, q=None):
        threading.Thread.__init__(self)
        self._threadID = self.__threadCount
        self.__threadCount += 1
        self._name = name
        self._queue = q

    def run(self):
        print("开始线程： " + self._name)
        # queueLock = threading.Lock()
        while not g_exitFlag:
            self.__lock.acquire()
            # queueLock.acquire()
            if not self._queue.empty():
                data = self._queue.get()
                self.__lock.release()
                # queueLock.release()
                process_image(data)
            else:
                self.__lock.release()
                # queueLock.release()
        print("退出线程： " + self._name)
    @property
    def threadID(self):
        return self._threadID
        
def divide_conquer():
    # global image_path_prod, txt_path_prod, allpic, curImage, nowtxt, nowline, invalidimg
    os.chdir(os.path.join(os.getcwd(), IMAGE_TRAIN_PATH))   #修改当前工作路径, 方便获取文件名
    image_names_train = glob.glob('*.jpg')                     #获取工作路径下所有jpg格式文件名到list中
    count_img = len(image_names_train) 
    print("total images: {}".format(count_img))
    #划分任务分配给多线程
    threadCount = 2 * multiprocessing.cpu_count()
    threadNames = ['thread-{}'.format(i) for i in range(threadCount)]
    threads = []
    queueLock1 = threading.Lock()
    workQueue = queue.Queue(2*threadCount)
    #创建新线程
    for tName in threadNames:
        thread = myThread(tName, workQueue)
        thread.start()
        threads.append(thread)
    #分割数据      
    fraction_size = int(count_img/threadCount)
    remains_count = int(count_img % threadCount)
    fractions = []
    for i in range(threadCount):
        fractions.append(image_names_train[i*fraction_size:(i+1)*fraction_size])
    if remains_count:
        fractions.append(image_names_train[count_img - remains_count: count_img])
    #填充队列
    queueLock1.acquire()
    for each in fractions:
        workQueue.put(each)
    queueLock1.release()
    #等待队列清空
    while not workQueue.empty():
        pass
    #通知线程退出
    global g_exitFlag   #全局变量在函数内部引用没有歧义， 但是在函数内部修改值的时候，需要加上global声明，否则变为局部变量
    g_exitFlag = 1
    #等待所有线程结束
    for t in threads:
        t.join()
    print("图片分割结束")

def process_image(imageNames):
    #读取txt文件， 把每一行文本内容保存的新文件， 读取每行坐标调用裁剪函数
    imgCounts = len(imageNames)
    invalidimg = []
    for j in range(imgCounts):
        imageTxt = os.path.join(g_txt_train_path, imageNames[j][:-4] + '.txt')     # txt路径
        print('处理图片: ' + imageTxt)
        imageName =imageNames[j]
        # curImage = imageName
        # nowtxt = imageTxt
        # nowline = 0
        imgSrc = cv2.imread(imageName)
        if(imgSrc is None):
            invalidimg.append(imageName)
        else:
            F = open(imageTxt,'rb')								#以二进制模式打开目标txt文件
            lines = F.readlines()								#逐行读入内容
            length=len(lines)
            s = 0                                               #计算图片编号，对应文本描述
            for i in range(length):
                lines[i] = str(lines[i], encoding = "utf-8")    #从bytes转为str格式
                lineContent = lines[i].split(',')[-1:]
                # nowline = i
                if ((lineContent != ['###\n']) and (lineContent != ['###'])):
                    s = s + 1
                    # allpic+=1
                    #保存新图片/txt格式为"原名字+编号+.jpg/.txt"
                    newImageName = os.path.join(g_img_train_prod_path, imageName[:-3] + str(s) + '.jpg')
                    newTxtName = os.path.join(g_txt_train_prod_path, imageName[:-3] + str(s) + '.txt')
                    #写入新TXT文件
                    if (s == length):
                        lineContent = str(lineContent)[2:-2]
                    else:
                        lineContent = str(lineContent)[2:-4]
                    file = open(newTxtName,'w')				#打开or创建一个新的txt文件
                    file.write(lineContent)        					#写入内容信息  
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
                    rotate(imgSrc,pt1,pt2,pt3,pt4,newImageName)     #计算旋转角度并截取图片  
    
'''旋转图像并剪裁'''
def rotate(img,                    # 图片
           pt1, pt2, pt3, pt4,     # 四点坐标
           newImageName):            # 输出图片路径
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
        cv2.imwrite(newImageName, imgOut)  # 保存得到的旋转后的矩形框
        return imgRotation                 # rotated image

    
if __name__=="__main__":
    cur_path = os.getcwd()
    print("cur_path: {}".format(cur_path))
    # image_path_prod = os.path.join(cur_path, "dataset/img_train_prod")
    # txt_path_prod = os.path.join(cur_path, "dataset/txt_train_prod")
    if not os.path.exists(g_img_train_prod_path):
        os.mkdir(g_img_train_prod_path)
    if not os.path.exists(g_txt_train_prod_path):
        os.mkdir(g_txt_train_prod_path)
    # allpic = 0
    # curImage = ''
    # nowtxt = ''
    # nowline = 0
    # invalidimg=[]
    # directory = os.path.join(cur_path, TXT_TRAIN_PATH) #TXT文件路径
    # ext = '.txt'
    divide_conquer()


