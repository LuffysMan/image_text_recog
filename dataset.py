#coding=utf-8
import cv2
import math
import os
import numpy as np
import glob
import pandas as pd 
import functools

import threading
import queue
import multiprocessing

from math import fabs, sin, cos, acos, radians

IMAGE_TRAIN_PATH = 'dataset/image_train'
TXT_TRAIN_PATH = 'dataset/txt_train'
IMAGE_TRAIN_PROD_PATH = 'dataset/img_train_prod'     #预处理后的图像路径
TXT_TRAIN_PROD_PATH = 'dataset/txt_train_prod'       #预处理后的图像对应文本路径
RECORD_PATH = 'dataset/record'
g_img_train_path = os.path.join(os.getcwd(), IMAGE_TRAIN_PATH)
g_txt_train_path = os.path.join(os.getcwd(), TXT_TRAIN_PATH)
g_img_train_prod_path = os.path.join(os.getcwd(), IMAGE_TRAIN_PROD_PATH)
g_txt_train_prod_path = os.path.join(os.getcwd(), TXT_TRAIN_PROD_PATH)
g_record_path = os.path.join(os.getcwd(), RECORD_PATH)

# g_exitFlag_workqueue = 0
# g_exitFlag_recQueue = 0

g_thread_count = multiprocessing.cpu_count()   #开辟线程数量
g_img_total = 0             #图片总数
g_img_prod_count = 0        #已处理图片总数
g_img_count_lock = threading.Lock()

recQueue = queue.Queue()         #创建容量为100的队列，用于接收record
recQueueLock = threading.Lock()  
workQueue = queue.Queue(2*g_thread_count)   #用户图像裁剪的任务队列锁 
cropQueueLock = threading.Lock()              

class myThread(threading.Thread):
    __threadCount = 0
    # __lock = threading.Lock()
    def __init__(self, name, function=None, queue=None, lock=None):
        #输入参数：线程名称， 线程函数， 线程函数处理的队列， 线程锁
        threading.Thread.__init__(self, name=name)
        self._threadID = self.__threadCount
        self.__threadCount += 1
        # self._name = name
        self._func = function
        self._queue = queue
        self.__lock = lock
        self._exitflag = 0

    def run(self):
        # print("开始线程： " + self._name)
        # queueLock = threading.Lock()
        while not self._exitflag:
            self.__lock.acquire()
            # queueLock.acquire()
            if not self._queue.empty():
                data = self._queue.get()
                self.__lock.release()
                self._func(data)
            else:
                self.__lock.release()
                # queueLock.release()
        # print("退出线程： {}".format(self._name))
    @property
    def threadID(self):
        return self._threadID
    def exit(self):
        self._exitflag = 1



def log(text=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            if text is not None:
                print( '%s %s():' % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator

@log()
def divide_conquer():
    global g_img_total, g_thread_count, cropQueueLock, workQueue
    os.chdir(os.path.join(os.getcwd(), IMAGE_TRAIN_PATH))       #修改当前工作路径, 方便获取文件名
    image_names_train = glob.glob('*.jpg')                     #获取工作路径下所有jpg格式文件名到list中
    g_img_total = len(image_names_train) 
    print("total images: {}".format(g_img_total))
    #划分任务分配给多线程
    threadNames = ['thread-crop{}'.format(i) for i in range(g_thread_count)]
    threads = []
    #创建新线程
    print("start {} threads".format(g_thread_count))
    for tName in threadNames:
        thread = myThread(tName, cropping_image, workQueue, cropQueueLock)
        thread.start()
        threads.append(thread)
    #分割数据      
    fraction_size = int(g_img_total/g_thread_count)
    remains_count = int(g_img_total % g_thread_count)
    fractions = []
    for i in range(g_thread_count):
        fractions.append(image_names_train[i*fraction_size:(i+1)*fraction_size])
    if remains_count:
        fractions.append(image_names_train[g_img_total - remains_count: g_img_total])
    #填充队列
    cropQueueLock.acquire()
    for each in fractions:
        workQueue.put(each)
    cropQueueLock.release()
    #等待队列清空
    while not workQueue.empty():
        pass
    #通知线程退出
    # global g_exitFlag_workqueue   #全局变量在函数内部引用没有歧义， 但是在函数内部修改值的时候，需要加上global声明，否则变为局部变量
    # g_exitFlag_workqueue = 1
    #开启record处理线程
    recThreads = processing_record()

    #通知线程退出并等待所有线程结束
    for t in threads:
        print("exit:", t.getName())
        t.exit()
        t.join()
    print("图片分割结束")
    for t in recThreads:
        print("exit:", t.getName())
        t.exit()
        t.join()
    print("record处理结束")
    #图片分割结束， 停止向recQueue填充， 通知record处理线程退出
    # global g_exitFlag_recQueue
    # g_exitFlag_recQueue = 1

@log('log')
def cropping_image(imageNames):
    #读取txt文件， 把每一行文本内容保存的新文件， 读取每行坐标调用裁剪函数
    imgCounts = len(imageNames)
    invalidimg = []
    global g_img_prod_count
    for j in range(imgCounts):
        g_img_count_lock.acquire()
        g_img_prod_count += 1
        g_img_count_lock.release()
        print("\r{}/{}".format(*(str(g_img_prod_count), str(g_img_total))), end='', flush=True)      #实时输出处理进度
        imageTxt = os.path.join(g_txt_train_path, imageNames[j][:-4] + '.txt')     # txt路径
        imageName =imageNames[j]
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
                if ((lineContent != ['###\n']) and (lineContent != ['###'])):
                    s = s + 1
                    #保存新图片/txt格式为"原名字+编号+.jpg/.txt"
                    newImageName = os.path.join(g_img_train_prod_path, imageName[:-3] + str(s) + '.jpg')
                    # newTxtName = os.path.join(g_txt_train_prod_path, imageName[:-3] + str(s) + '.txt')
                    #写入新TXT文件
                    if (s == length):
                        lineContent = str(lineContent)[2:-2]
                    else:
                        lineContent = str(lineContent)[2:-4]
                    # file = open(newTxtName,'w', encoding='utf-8)		#打开or创建一个新的txt文件
                    # file.write(lineContent)        					#写入内容信息  
                    # file.close()  
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
                    imgOut = rotate(imgSrc,pt1,pt2,pt3,pt4,newImageName)     #计算旋转角度并截取图片 
                    generate_record(lineContent, imgOut) 
    
'''旋转图像并剪裁'''
@log()
def rotate(img,                    # 图片
           pt1, pt2, pt3, pt4,     # 四点坐标
           newImageName):            # 输出图片路径
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)      # 矩形框的宽度
    #heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
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
        # cv2.imwrite(newImageName, imgOut)  # 保存得到的旋转后的矩形框
        # return imgRotation                 # rotated image
        return imgOut

@log()
def generate_record(lineContent, imgOut):
    #将图像和标签打包成record, 压入队列， 供写文件线程读取
    # imgOut = np.array(imgOut, dtype=np.uint8)   #以uint8格式存储， 读取的时候按此格式恢复
    global recQueue, recQueueLock       #使用全局变量需要声明
    record = {
        'label': [lineContent],
        'height': [imgOut.shape[0]],
        'width':[imgOut.shape[1]],
        'channels': [imgOut.shape[2]],
        'image': [imgOut.reshape((1,-1))],
    }
    #将打包好的record压入队列中
    recQueueLock.acquire()      
    recQueue.put(record)
    recQueueLock.release()

# @log('log')
def write_record(record):
    #record处理函数， 将record追加写入到对应的文件中
    df = pd.DataFrame(record)
    tName = threading.current_thread().getName()
    path = os.path.join(g_record_path, tName)
    df.to_csv(path, mode='a', header=None, sep=',', index=None)    #以线程名为文件名， 每个线程写自己的文件

def processing_record():
    global recQueue, recQueueLock
    #创建线程
    tNames = ["thread_record-{}".format(i) for i in range(12)]
    threads = []
    for tName in tNames:
        thread = myThread(tName, function=write_record, queue=recQueue, lock=recQueueLock)
        thread.start()
        threads.append(thread)
    return threads
    # #等待所有线程结束
    # for t in threads:
    #     t.join()
    # print("记录保存完成")


    
if __name__=="__main__":
    # cur_path = os.getcwd()
    # print("cur_path: {}".format(cur_path))
    # image_path_prod = os.path.join(cur_path, "dataset/img_train_prod")
    # txt_path_prod = os.path.join(cur_path, "dataset/txt_train_prod")
    if not os.path.exists(g_img_train_prod_path):
        os.mkdir(g_img_train_prod_path)
    if not os.path.exists(g_txt_train_prod_path):
        os.mkdir(g_txt_train_prod_path)
    if not os.path.exists(g_record_path):
        os.mkdir(g_record_path)
    #通过划分图像文件， 并行的进行裁剪处理
    divide_conquer()
    #将裁剪好的图片和label写入record文件
    # processing_record()



