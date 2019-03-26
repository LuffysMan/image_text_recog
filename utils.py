#coding: utf-8
import threading
import functools

class myThread(threading.Thread):
    __threadCount = 0
    # __lock = threading.Lock()
    def __init__(self, name, function, queue, lock):
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

















