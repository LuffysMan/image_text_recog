# image_text_recog
MTWI 2018 挑战赛一 网络图像文本识别
任务初步划分:
1. 数据预处理: 包括处理图像尺寸不一致的问题, 扩充图像训练集, 噪声处理等;
2. 模型定义: 定义学习模型的各种超参数, 比如神经网络的层数, 各层的节点数, 激活函数, 前向传播过程的定义, 损失函数的定义等;
3. 模型可视化: 包括模型训练loss, acc, 模型验证loss, acc的曲线绘制; 也可以考虑模型各层的可视化, 卷积核的可视化等, 可帮助加深对模型理解和调试;
4. 算法调试和优化: 对定义好的算法进行各部分进行允许调试, 查找不合理或者有误的部分, 提出优化的意见;

#数据集预处理部分
dataset.py
dataset|
        |image_train        #训练集图片
        |txt_train          #训练集图片对应文本
        |image_train_prod   #训练集图片进行分割后结果
        |txt_train_prod     #训练集文本处理后结果
        |image_test         #测试集图片
##关于dataset.py
需要加载的模块：opencv和math
主要功能：读取image_train和txt_train中的数据集， 根据提供的坐标对图片进行裁剪， 并输出对应的图片和文本
到img_train_prod和txt_train_prod； 
使用方法：
运行前可配置开启的线程数（默认线程数同计算机CPU数量)， 配置变量g_thread_count， 建议数量不超过cpu数量2倍
在终端输入：python dataset.py

参考文献:
大家各自进行研究和实现的过程中, 找到的比较有用的文章等资料, 把题目和链接写到对应的类别下面, 方便大家互相扩充知识面;
1.数据预处理:
python扩大训练集样本数量-图片转换、改变尺寸 https://blog.csdn.net/weixin_42052460/article/details/80861056

【python】详解zipfile模块读取处理压缩文件实例: https://blog.csdn.net/brucewong0516/article/details/79064384

多线程处理图片：
    Python 类中的"静态"成员变量: https://www.cnblogs.com/turtle-fly/p/3280610.html

    Python的访问修饰符： http://blog.sina.com.cn/s/blog_bb48e6be0102wbgd.html

    使用@property: 廖雪峰博客

    python 全局变量引用与修改： https://www.cnblogs.com/yanfengt/p/6305542.html


2.模型定义:

3.模型可视化:

4. 算法调试和优化:






