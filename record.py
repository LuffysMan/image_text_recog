#coding=utf-8
import cv2
import math
import os
import numpy as np
import glob
# import pandas as pd 
import json

import threading
import queue
import multiprocessing

from math import fabs, sin, cos, acos, radians
from utils import myThread, log, chdir
from parameters import IMAGE_TRAIN_PATH, TXT_TRAIN_PATH, BATCH_SIZE            

np.set_printoptions(threshold=1000000000)

g_thread_count = multiprocessing.cpu_count()   #开辟线程数量
g_img_total = 0             #图片总数
g_img_prod_count = 0        #已处理图片总数
g_img_count_lock = threading.Lock()
g_active_cropThread_Count = 0    #裁剪工作线程计数

recQueue = queue.Queue()         #用于接收record
recQueueLock = threading.Lock()  
workQueue = queue.Queue(2*g_thread_count)   #用户图像裁剪的任务队列 
cropQueueLock = threading.Lock()              

@log()
def divide_conquer(image_names_train):
    global g_img_total, g_thread_count, cropQueueLock, workQueue, g_active_cropThread_Count
    # with chdir(IMAGE_TRAIN_PATH) as ch:
    #     # os.chdir(os.path.join(os.getcwd(), IMAGE_TRAIN_PATH))       #修改当前工作路径, 方便获取文件名
    #     image_names_train = glob.glob('*.jpg')                     #获取工作路径下所有jpg格式文件名到list中
    #     # image_names_train = glob.glob(os.path.join(IMAGE_TRAIN_PATH, '*.jpg'))                     #获取工作路径下所有jpg格式文件名到list中
    g_img_total = len(image_names_train)
    print("total images: {}".format(g_img_total))
    #划分任务分配给多线程
    threadNames = ['thread-crop{}'.format(i) for i in range(g_thread_count)]
    threads = []
    #创建新线程
    print("start {} threads".format(g_thread_count))
    for tName in threadNames:
        thread = myThread(tName, t_crop_image, workQueue, cropQueueLock)
        thread.start()
        g_active_cropThread_Count += 1
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
    print("start filling cropping queue")
    cropQueueLock.acquire()
    for each in fractions:
        workQueue.put(each)
    cropQueueLock.release()

    #通知线程退出
    exit_crop_thread = threading.Thread(target = wait_exit_threads, name="wait", args=(threads, workQueue))
    exit_crop_thread.start()
    # exit_thread.join()
    # print("test join")

def wait_exit_threads(threads, workQueue):
    global g_active_cropThread_Count
    #等待图像裁剪任务队列被清空
    while not workQueue.empty():
        pass
    #通知裁剪任务处理线程退出并等待所有线程结束
    for t in threads:
        print("\r\nexit:", t.getName())
        t.exit()
        t.join()
        g_active_cropThread_Count -= 1

@log()
def t_crop_image(imageNames):
    #读取txt文件， 把每一行文本内容保存的新文件， 读取每行坐标调用裁剪函数
    imgCounts = len(imageNames)
    invalidimg = []
    records = {}
    tName = threading.current_thread().getName()
    for j in range(imgCounts):
        # tmpName = imageNames[j].split('/')[-1]
        # tmpName = tmpName.split('.')[-3:-1]
        # tmpName.append('txt')
        # print(tmpName)
        # imageTxt = os.path.join(TXT_TRAIN_PATH, '.'.join(tmpName))     # txt路径
        imageTxt = os.path.join(TXT_TRAIN_PATH, imageNames[j][:-4] + '.txt')     # txt路径
        imageName =os.path.join(IMAGE_TRAIN_PATH, imageNames[j])
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
                    # newImageName = os.path.join(IMAGE_TRAIN_PROD_PATH, imageName[:-3] + str(s) + '.jpg')
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
                    imgOut = rotate(imgSrc,pt1,pt2,pt3,pt4)     #计算旋转角度并截取图片 
                    # imgOut = rotate(imgSrc,pt1,pt2,pt3,pt4,newImageName)     #计算旋转角度并截取图片 
                    idx = "{}-{}-{}".format(tName, j, i)
                    records = generate_records(records, lineContent, imgOut, idx) 
    #最后剩余的record一次性入队
    if len(records) > 0:
        recQueueLock.acquire()      
        recQueue.put(records)
        recQueueLock.release()

    
'''旋转图像并剪裁'''
@log()
def rotate(img, pt1, pt2, pt3, pt4):   # 图片, 四点坐标, 输出图片路径
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
def generate_records(records, lineContent, imgOut, idx):
    #将图像和标签打包成record, 压入队列， 供写文件线程读取
    # imgOut = np.array(imgOut, dtype=np.uint8)   #以uint8格式存储， 读取的时候按此格式恢复
    global recQueue, recQueueLock       #使用全局变量需要声明
    try:
        record = {
            'L': lineContent,
            'H': imgOut.shape[0],
            'W': imgOut.shape[1],
            'C': imgOut.shape[2],
            'I': imgOut.reshape((1,-1)).tolist(),
            }
        records[idx] = record 
    except:     #遇到异常的数据直接跳过
        pass
    finally:
        pass

    #将打包好的record压入队列中
    # if len(records['height']) >= 100:
    if len(records) >= BATCH_SIZE:
        recQueueLock.acquire()      
        recQueue.put(records)
        recQueueLock.release()
        records = {}
    return records

# # @log('log')
# def t_write_record(records):
#     global g_img_prod_count
#     g_img_count_lock.acquire()
#     g_img_prod_count += len(records)
#     g_img_count_lock.release()
#     # print("\r{}--{}--{}".format(*(str(g_img_prod_count), str(recQueue.qsize()), str(g_img_total))), end='', flush=True)      #实时输出处理进度
#     #record处理函数， 将record追加写入到对应的文件中
#     jsonRecords = json.dumps(records)
#     tName = threading.current_thread().getName()
#     path = os.path.join(RECORD_PATH, tName)
#     with open(path, mode='a') as f:
#         f.write(jsonRecords)
#         f.write('\r\n')
#     # df = pd.DataFrame(records)
#     # df.to_csv(path, mode='a', header=None, sep=',', index=None)    #以线程名为文件名， 每个线程写自己的文件

# def processing_record():
#     global recQueue, recQueueLock
#     #创建线程
#     tNames = ["thread_record-{}".format(i) for i in range(12)]
#     threads = []
#     for tName in tNames:
#         thread = myThread(tName, function=t_write_record, queue=recQueue, lock=recQueueLock)
#         thread.start()
#         threads.append(thread)
#     return threads
#     # #等待所有线程结束
#     # for t in threads:
#     #     t.join()
#     # print("记录保存完成")


# def start_produce():
#     #开始读取图片并裁剪, 将结果压入队列, 使用是从队头取
#     # if not os.path.exists(path):
#         # os.mkdir(path)
#     #通过划分图像文件， 并行的进行裁剪处理
#     divide_conquer()

def get_cropThreadCount():
    global g_active_cropThread_Count
    return g_active_cropThread_Count




