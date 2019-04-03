# image_text_recog
MTWI 2018 挑战赛一 网络图像文本识别
任务初步划分:
1. 数据预处理: 包括处理图像尺寸不一致的问题, 扩充图像训练集, 噪声处理等;
2. 模型定义: 定义学习模型的各种超参数, 比如神经网络的层数, 各层的节点数, 激活函数, 前向传播过程的定义, 损失函数的定义等;
3. 模型可视化: 包括模型训练loss, acc, 模型验证loss, acc的曲线绘制; 也可以考虑模型各层的可视化, 卷积核的可视化等, 可帮助加深对模型理解和调试;
4. 算法调试和优化: 对定义好的算法进行各部分进行允许调试, 查找不合理或者有误的部分, 提出优化的意见;

# 数据集预处理部分(崔悦)
dataset.py
dataset|
        |image_train        #训练集图片
        |txt_train          #训练集图片对应文本
        <!-- |image_train_prod   #训练集图片进行分割后结果 -->
        <!-- |txt_train_prod     #训练集文本处理后结果 -->
        |image_test         #测试集图片
训练样例总数：142434
## 关于dataset.py
<!-- 需要加载的模块：opencv和math
主要功能：读取image_train和txt_train中的数据集， 根据提供的坐标对图片进行裁剪， 并输出对应的图片和文本
到img_train_prod和txt_train_prod； 
使用方法：
运行前可配置开启的线程数（默认线程数同计算机CPU数量)， 配置变量g_thread_count， 建议数量不超过cpu数量2倍
在终端输入：python dataset.py -->
提供的接口:
1. train_valid_split()      #功能: 按比例随机划分训练集和验证集, 输入: 图片所在路径和划分比例; 输出: 训练集和测试集文件名列表
2. read_data_sets()       
功能: 返回一个DataSets对象, 启动图片处理线程
输入: 训练集或者验证集文件名列表
输出: DataSets对象
3. DataSets.next_batch()
功能: 返回BATCH_SIZE大小的数据集
输入: 无
输出: 两个list: images, labels



使用方法:
具体使用方法参考dsatasetEx.py中的demo函数
注意事项: 在window下跑, 需要 在parameters.py中修改 路径字符串中的'/'为'\\',

## 关于输入图像尺寸不同的处理办法
<!-- - 方案1： 将图像按照给定的bounding box进行分割， 并分批存储到tfrecord，  
进行神经网络训练前，先读取tfrecord， 然后将图像还原， 进一步resize为统一的高度， 输入到crnn  -->
<!-- - 由于tensorflow与paddlepaddle不兼容， 所以将record做成与框架无关， 采用pandas追加式写txt -->
1. 读取原始图片和对应txt文件
2. 根据文本对应坐标裁剪图片
3. 统一裁剪后的图片尺寸(默认290x32)

## 如何计算loss
- 图像对应的文本如何进行编码？ 训练模型如何计算loss
文本中的每个字符按照字典序编码(字典需自行构造)， loss计算使用”编辑距离“

## 参考文献:  
1. 图像预处理
  python扩大训练集样本数量-图片转换、改变尺寸 https://blog.csdn.net/weixin_42052460/article/details/80861056 
  在Python and OpenCV中做图象处理:改变大小，旋转和裁剪(翻译)  https://blog.csdn.net/fxt570762000/article/details/80241446 
  图像处理之PIL.Image与numpy.array之间的相互转换 https://blog.csdn.net/qq_30159015/article/details/80070514
  第一篇 Python图片处理模块PIL（pillow） http://www.cnblogs.com/chimeiwangliang/p/7130434.html
  Python用Pillow(PIL)进行简单的图像操作 https://www.cnblogs.com/sun-haiyu/p/7127582.html
2. 压缩文件处理
  【python】详解zipfile模块读取处理压缩文件实例: https://blog.csdn.net/brucewong0516/article/details/79064384  
3. 多线程处理图片：  
    Python 类中的"静态"成员变量: https://www.cnblogs.com/turtle-fly/p/3280610.html  
    Python的访问修饰符： http://blog.sina.com.cn/s/blog_bb48e6be0102wbgd.html  
    使用@property: 廖雪峰博客  
    python 全局变量引用与修改： https://www.cnblogs.com/yanfengt/p/6305542.html  
4. 构建字典， 处理图像对应的字符标签
  超酷算法（1）：BK树(http://blog.jobbole.com/78811/)
  文字识别(OCR)CRNN（基于pytorch、python3） 实现不定长中文字符识别(https://blog.csdn.net/Sierkinhane/article/details/82857572)
## 遇到的问题
问题1: 在使用pandas.to_csv()函数将图像数据存储到txt中的时候, 出现了部分图像数据变为省略号的情况
原因: 图像原始数据是numpy数组, numpy数组在使用print函数输出的时候, 如果超过1000个元素, 会用省略号'...'来代替部分元素; 并且实际情况
是numpy数组作为列表元素或者转为str的时候, 也用'...'代替的部分元素, 导致写入txt时也是省略号.
比如: 
    a = np.arange(1000)
    b = str(a) 
    c = [a]
    print(b, c)     #都是缩写的情况
解决: 方案1 使用 np.tostring()或者np.tolist(), 比如 b=a.tolist()
方案2 在代码开头设置 np.set_printoptions(threshold=1000*1000*1000) #这样一百万个元素以下的数组都可以正常显示或写入
# 模型定义(肖扬&汤凌风):

# 模型可视化(梁帆):

# 算法调试和优化(李昊博):






