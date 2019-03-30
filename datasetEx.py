#coding: utf-8
'''
预处理数据, 封装成方便使用的数据集
提供随机batch功能(采用生产者消费者模式， 进行数据语预取， 随机出队列)
提供统一高度的图像, 作为crnn的输入
构建字库, 对label进行编码
记录log
'''
# import pandas as pd
import numpy as np 
# import codecs
import os
import queue
import threading 
# import json


from utils import myThread, log
from parameters import  RECORD_PATH, IMAGE_TRAIN_PATH, TXT_TRAIN_PATH, BATCH_SIZE
from record import recQueue, recQueueLock, divide_conquer, get_cropThreadCount

# recQueue = queue.Queue(2)       #最大容量为2, nextbatch()读一个往里面放一个
# recQueueLock = threading.Lock()
# fileQueue = queue.Queue()
# fileQueueLock = threading.Lock()

class chdir():
    def __init__(self, newdir):
        self._olddir = os.getcwd()
        self._newdir = newdir
    def __enter__(self):
        os.chdir(self._newdir)
        # print("enter work dir", self._newdir)
    def __exit__(self, a, b, c):
        os.chdir(self._olddir)
        # print("exit work dir ", self._newdir)


class DataSet(object):
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
    def __init__(self):
        self.__start_produce()

    def __start_produce(self):
        #启动图像裁剪线程
        divide_conquer()

    def next_batch(self):
        #从工作队列recQueue取出裁剪好的图像和对应label, 大小为BATCH_SIZE, 定义在parameters.py
        images, labels = self.train.read_record()
        while not images and not labels:
            if 0 == get_cropThreadCount():
                return {}, {}
            images, labels = self.train.read_record()
        return images, labels

@log()
def read_data_sets():
    data_sets = DataSets()
    data_sets.train = DataSet(recQueue, recQueueLock, epochs=1)
    return data_sets

# def next_batch(data_sets):
#     images, labels = data_sets.train.read_record()
#     while not images and not labels:
#         if 0 == get_cropThreadCount():
#             return {}, {}
#         images, labels = data_sets.train.read_record()
#     return images, labels

if __name__ == "__main__":
    # start_produce()
    data_sets = read_data_sets()
    step = 0
    while True:
        images, labels = data_sets.next_batch()
        if images and labels:
            print(step, len(images), len(labels))  #可用于训练, images需要将height统一, labels需要进行编码
            step += 1
        else:
            print("over")
            break











