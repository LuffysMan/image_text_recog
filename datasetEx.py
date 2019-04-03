#coding: utf-8
'''
预处理数据, 封装成方便使用的数据集
提供随机batch功能(采用生产者消费者模式， 进行数据语预取， 随机出队列)
提供统一高度的图像, 作为crnn的输入; 图像标准化(暂时不确定, 没有进行标准化)
构建字库, 对label进行编码(未实现)
记录log(未实现)
'''
# import pandas as pd
import numpy as np 
# import codecs
import os
import queue
import threading 
import random
import glob
import time

from PIL import Image

from utils import myThread, log, chdir
from parameters import  RECORD_PATH, IMAGE_TRAIN_PATH, TXT_TRAIN_PATH, BATCH_SIZE
from record import recQueue, recQueueLock, divide_conquer, get_cropThreadCount

# recQueue = queue.Queue(2)       #最大容量为2, nextbatch()读一个往里面放一个
# recQueueLock = threading.Lock()
# fileQueue = queue.Queue()
# fileQueueLock = threading.Lock()

# class chdir():
#     def __init__(self, newdir):
#         self._olddir = os.getcwd()
#         self._newdir = newdir
#     def __enter__(self):
#         os.chdir(self._newdir)
#         # print("enter work dir", self._newdir)
#     def __exit__(self, a, b, c):
#         os.chdir(self._olddir)
#         # print("exit work dir ", self._newdir)


class Consumer(object):
    @log('call: ')
    def __init__(self, recQueue, recQueueLock, epochs=1):
        # self._recFilePath = recFilePath
        self._inputQueue = recQueue     #最多同时加载两个record       
        self._inputQueueLock = recQueueLock
        self._epochs = epochs               #数据循环的次数
        # self._batch_size = batch_size

        self._num_examples = 0
        # self._images = None
        # self._labels = None
        # self._steps = self._epochs * len(os.listdir(recFilePath))
        # self._index_in_epoch = 0
        # self._is_epochs_finished = False        #读取record文件已结束

    @log()
    def read_record(self):
        #从输入队列读取records
        images = []
        labels = []
        # while self._inputQueue.empty():
        #     #等待的输入队列被填充
        #     if g_active_cropThread_Count == 0:
        #         return
        #     pass
        self._inputQueueLock.acquire()
        try:
            records = self._inputQueue.get(block=False)
        except:
            records = {}
        finally:
            self._inputQueueLock.release()
        # for line in lines:
        # records = json.loads(data)
        for key in records:
            record = records[key]
            image_raw = record['I']
            H = record['H']
            W = record['W']
            C = record['C']
            image = np.array(image_raw).reshape((H,W,C))
            images.append(image)
            labels.append(record['L'])
        return images, labels

    # @log('call: ')
    # def next_batch(self):
    #     """Return the next 'batch_size' data from this data set."""
    #     return self.__read_record()


class DataSets(object):
    def __init__(self, filenames):
        self._target_height = 32       #将图像高度统一为32个像素
        self._target_width = 290       ##经统计, 90%的图片按比例将高度缩放至32时, 宽度不超过290,  80%的图片按比例将高度缩放至32时, 宽度不超过203,
        self._box = (0,0,self._target_width, self._target_height)
        # self._train_test_ratio = 0.8
        # self._datapath = datapath
        self._image_files = filenames
        # self._valid_images = []
        # self.train_valid_split()
        self.__start_produce()

    def __start_produce(self):
        #启动图像裁剪线程
        divide_conquer(self._image_files)

    def next_batch(self):
        #从工作队列recQueue取出裁剪好的图像和对应label, 大小为BATCH_SIZE, 定义在parameters.py
        self._images, self._labels = self.train.read_record()
        while not  self._images and not self._labels:
            if 0 == get_cropThreadCount():      #查询是否已经停止裁剪图像
                return {}, {}
            self._images, self._labels = self.train.read_record()
        # return self._images, self._labels
        # self.writeimage(self._images, self._labels)
        return self.resize_with_crop_pad(self._images, self._labels)

    def resize_with_crop_pad(self, images, labels):
        result_images = []
        result_labels = []
        # images = self._images
        #调整图像为统一高度, 满足crnn需要
        i = 0 
        bad = []
        for image in images:
            try:
                H = image.shape[0]
                W = image.shape[1]
                ratioW = self._target_width/W          #经统计, 90%的图片按比例将高度缩放至32时, 宽度不超过290 
                ratioH = self._target_height/H
                if ratioH <= ratioW:
                    size = (int(W*ratioH), 32)
                else:
                    size = (290, int(H*ratioW))
                # ratio = self._target_height/H
                im = Image.fromarray(image.astype('uint8')).convert('RGB')
                im = im.resize(size, Image.BILINEAR)    #将图像缩放至(<290, 32) 或(290, <32)
                im = im.crop(self._box)                 #填充图像, 使之为(290, 32)
                result_images.append(np.array(im))
                result_labels.append(labels[i])
            except:
                print("failed resize", image.shape)
                # im.save('./test/resized/%s-%.4d.jpg'%(time.time(), i))
                bad.append(i)
            finally:
                i += 1
        return result_images, result_labels

    # def writeimage(self, images, labels):
    #     i = 0
    #     for image in images:
    #         im = Image.fromarray(image.astype('uint8')).convert('RGB')
    #         im.save('./test/origin/%s-%.4d.jpg'%(time.time(), i))
    #         i += 1
@log()
def read_data_sets(filenames):
    data_sets = DataSets(filenames)
    data_sets.train = Consumer(recQueue, recQueueLock, epochs=1)
    return data_sets

# def next_batch(data_sets):
#     images, labels = data_sets.train.read_record()
#     while not images and not labels:
#         if 0 == get_cropThreadCount():
#             return {}, {}
#         images, labels = data_sets.train.read_record()
#     return images, labels


def train_valid_split(datapath, ratio=0.8, shuffle=True):
    with chdir(datapath) as ch:
        # os.chdir(os.path.join(os.getcwd(), IMAGE_TRAIN_PATH))       #修改当前工作路径, 方便获取文件名
        image_names_train = glob.glob('*.jpg')                     #获取工作路径下所有jpg格式文件名到list中
        # image_names_train = glob.glob(os.path.join(IMAGE_TRAIN_PATH, '*.jpg')) 
    #将数据集分割为训练集和验证集
    random.shuffle(image_names_train)
    mid = int(ratio*len(image_names_train))
    train_image_files = image_names_train[0: mid]
    valid_image_files = image_names_train[mid: ]
    return train_image_files, valid_image_files
    
def demo():
    #首先划分训练集和验证集
    train_image_files, valid_image_files = train_valid_split(IMAGE_TRAIN_PATH, ratio=0.7)
    print(len(train_image_files))
    print('start trainning')
    data_sets = read_data_sets(train_image_files)                    #开始读取图像数据
    step = 0
    #读取训练集并训练
    while True:
        images, labels = data_sets.next_batch()
        if images and labels:                       #如果为空, 表示数据已经循环一次
            #train()        #训练模型
            print("train batch: ", len(images), len(labels))
            step += 1
        else:
            print("over")
            break
    #读取验证集并验证
    print('start validating')
    data_sets = read_data_sets(valid_image_files)                    #开始读取图像数据
    print(len(valid_image_files))
    step = 0
    while True:
        images_valid, labels_valid = data_sets.next_batch()
        if images_valid and labels_valid:                       #如果为空, 表示数据已经循环一次
            #train()        #训练模型
            print("valid batch: ", len(images_valid), len(labels_valid))
            step += 1
        else:
            print("over")
            break


if __name__ == "__main__":
    demo()









