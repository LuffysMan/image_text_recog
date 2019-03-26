#coding: utf-8
'''
预处理数据, 封装成方便使用的数据集
提供随机batch功能(采用生产者消费者模式， 进行数据语预取， 随机出队列)
提供图像随机翻转的功能
构建字库
记录log
'''
import pandas as pd
import numpy as np 
# import codecs
import os
import queue
import threading 

from utils import myThread, log
from parameters import  RECORD_PATH, IMAGE_TRAIN_PATH, TXT_TRAIN_PATH

recQueue = queue.Queue(2)       #最大容量为2, nextbatch()读一个往里面放一个
recQueueLock = threading.Lock()
fileQueue = queue.Queue()
fileQueueLock = threading.Lock()

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

# def get_trainset():
#     with chdir(RECORD_PATH) as ch:
#         recFiles = os.listdir(RECORD_PATH)
#         recFile = next(recFiles)
#         return read_record(recFile)

# class Data_producer(object):
#     __queue = queue.Queue(2)        #record读取队列， 最多存储两个record
#     def __init__(self, *args, **kwargs):
#         pass


class DataSet(object):
    def __init__(self, recFilePath, epochs=1, batch_size = 32):
        self._recFilePath = recFilePath
        self._epochs = epochs               #数据循环的次数
        self._batch_size = batch_size

        self._num_examples = 0
        self._images = None
        self._labels = None
        self._steps = self._epochs * len(os.listdir(recFilePath))
        self._index_in_epoch = 0
        self._is_epochs_finished = False        #读取record文件已结束

        self._inputQueue = queue.Queue(2)     #最多同时加载两个record       
        self._inputQueueLock = threading.Lock()
        self.__init_producer()
    @log()
    def __init_producer(self):
        self._producer = Producer(self._recFilePath, self._inputQueue, self._inputQueueLock, self._epochs, numThreads=2)
        self._producer.start_produce()

    @log()
    def stop_produce(self):
        self._producer.stop_produce()

    @log()
    def fetch_data(self):
        #从image和label队列取数据, 如果没有通知record文件已经读取完毕, 则阻塞式读取
        # if not self._flag_finish_read_record: 
        assert not self._producer.is_finished or not self._inputQueue.empty()
        if not self._steps: 
            self._inputQueueLock.acquire() 
            (self._images, self._labels) = self._inputQueue.get() 
            self._inputQueueLock.release()
            self._steps -= 1
            self._num_examples = len(self._images) 
            return True
        else:
            (self._images, self._labels) = (None, None) 
            self._num_examples = 0 
            self.stop_produce()
            self._is_epochs_finished = True
            return False
        # elif not self._inputQueue.empty():
        #     self._inputQueueLock.acquire() 
        #     (self._images, self._labels) = self._inputQueue.get() 
        #     self._inputQueueLock.release() 
        #     self._num_examples = len(len(self._images)) 
        # else:
        #     self._images = None
        #     self._labels = None
        #     self._epochs_completed += 1

    @log()
    def next_batch(self):
        """Return the next 'batch_size' data from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += self._batch_size
        if self._index_in_epoch > self._num_examples:   
            #需要换一批新数据
            if not self.fetch_data():
                return None, None
                # self._producer.stop_produce()
                # return None             #迭代结束
            #Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            #Start next epoch
            start = 0
            self._index_in_epoch = self._batch_size
            assert self._batch_size < self._num_examples
        end = self._index_in_epoch
        return self._images[start: end], self._labels[start: end]
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def is_epochs_finished(self):
        return self._is_epochs_finished

class Producer(object):
    #生产者类, 生产(images, labels)供nextbatch()使用
    def __init__(self, recFilePath, outputQueue, outputQueueLock, epochs=1, numThreads=1):
        self._recFilePath = recFilePath
        self._outputQueue = outputQueue
        self._outputQueueLock = outputQueueLock
        self._epochs = epochs               #预期生产轮数  
        self._numThreads = numThreads   #生产者工作线程数量

        # self._epochs_completed = 0          #已完成的轮数
        # self._flag_stop_produce = False       #通知各线程停止生产
        self._threads = []
        self._fileQueue = queue.Queue()
        self._fileQueueLock = threading.Lock()

    @log()
    def __create_threads(self):
        # fileQueue = queue.Queue()
        # fileQueueLock = threading.Lock()
        #创建并启动线程, 处理record文件名队列
        tNames = ["producer-{}".format(i) for i in range(self._numThreads)]
        for tName in tNames:
            thread = myThread(tName, self.__read_record, self._fileQueue, self._fileQueueLock)
            thread.start()
            self._threads.append(thread)

    @log()
    def __fill_queue(self):
        #文件入队供producer生产用
        recFiles = os.listdir(self._recFilePath)
        self._fileQueueLock.acquire()
        for _ in range(self._epochs):
            for eachfile in recFiles:
                self._fileQueue.put(os.path.join(self._recFilePath, eachfile))
        self._fileQueueLock.release()

    @property
    def is_finished(self):
        #是否已经将任务队列任务都取走
        return self._fileQueue.empty()

    @log()
    def start_produce(self):
        self.__create_threads()
        self.__fill_queue()
        # self._epochs_completed += 1

    @log()
    def stop_produce(self):
        #结束生产
        while not self._fileQueue.empty():
            pass
        for t in self._threads:
            t.exit()
            t.join()
        # self._flag_stop_produce = True
        # return self._flag_stop_produce

    @log()
    def __read_record(self, filePath):
        #从指定路径读取record并解析
        df = pd.read_csv(filePath, header=None, sep=',')
        records = df.values.tolist()
        images = []
        labels = []
        for (label, H, W, C, image_raw) in records:
            try:
                label = label.strip('\n')
                image_raw = image_raw.strip('[]')
                image = np.array(image_raw.split())
                image =image.reshape((H,W,C))
                images.append(image)
                labels.append(label)
            except:
                print("parse error--", image.shape)

        self._outputQueueLock.acquire()
        self._outputQueue.put((images, labels))
        self._outputQueueLock.release()
        # return (images, labels)
    


    
    # #等待读取完毕, 关闭线程
    # while not fileQueue.empty():
    #     pass
    # for t in threads:
    #     t.exit()
    #     t.join()
    # g_flag_finish_read_record = 1

@log()
def read_data_sets():
    # global  recQueue, recQueueLock, fileQueue, fileQueueLock 
    class DataSets(object):
        pass
    data_sets = DataSets()
    # producer = Producer(fileQueue, fileQueueLock, epochs=1, numThreads=1)
    # producer.start_produce()
    # #文件入队共producer生产用
    # recFiles = os.listdir(RECORD_PATH)
    # fileQueueLock.acquire()
    # for eachfile in recFiles:
    #     fileQueue.put(os.path.join(RECORD_PATH, eachfile))
    # fileQueueLock.release()
    data_sets.train = DataSet(RECORD_PATH, epochs=1, batch_size=32)
    # data_sets.valid = DataSet(valid_images, valid_labels)
    return data_sets

if __name__ == "__main__":
    data_sets = read_data_sets()
    images, labels = data_sets.train.next_batch()
    print(images, labels)












