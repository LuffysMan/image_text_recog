#coding: utf-8
'''
预处理数据, 封装成方便使用的数据集
提供随机batch功能(采用生产者消费者模式， 进行数据语预取， 随机出队列)
提供图像随机翻转的功能
构建字库
记录log
'''
# import pandas as pd
import numpy as np 
# import codecs
import os
import queue
import threading 
import json


from utils import myThread, log
from parameters import  RECORD_PATH, IMAGE_TRAIN_PATH, TXT_TRAIN_PATH, BATCH_SIZE
from record import recQueue, recQueueLock, start_produce

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

    @log('call: ')
    def __read_record(self):
        #从输入队列读取records
        images = []
        labels = []
        while self._inputQueue.empty():
            #等待的输入队列被填充
            pass
        self._inputQueueLock.acquire()
        records = self._inputQueue.get(block=True)
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

    @log('call: ')
    def next_batch(self):
        """Return the next 'batch_size' data from this data set."""
        return self.__read_record()

@log()
def read_data_sets():
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(recQueue, recQueueLock, epochs=1)
    return data_sets

if __name__ == "__main__":
    start_produce()
    data_sets = read_data_sets()
    while True:
        images, labels = data_sets.train.next_batch()
        print(len(images), len(labels))












