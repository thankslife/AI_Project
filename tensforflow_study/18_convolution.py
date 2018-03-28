import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow as tf
import matplotlib.pyplot as plt


# 加载数据集
# 输入图片通常是3D，[height, width, channels]#channels通道数量
# mini-batch通常是4D，[mini-batch size, height, width, channels]
dataset = np.array(load_sample_images().images, dtype=np.float32) #images是两张图片
#dataset是4维的
# 数据集里面两张图片，一个中国庙宇，一个花
batch_size, height, width, channels = dataset.shape
print(batch_size, height, width, channels)

# 创建两个filter
# 高，宽，通道，卷积核
# 7, 7, channels, 2
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters_test[:, 3, :, 0] = 1  # 垂直 #高相当于行，宽相当于列 只有第三列是1 7*7的矩阵
filters_test[3, :, :, 1] = 1  # 水平

# filter参数是一个filters的集合
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
# strides=[1, 2, 2, 1] 中第一最后一个为1，中间对应sh和sw
# strides相当于步伐 sh 代表纵向的步伐 sw代表横向的步伐
convolution = tf.nn.conv2d(X, filter=filters_test, strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})#运行完的featchure_map

plt.imshow(output[0, :, :, 0])  # 绘制第一个图(原始的图)的第一个特征图
# plt.imshow(output[0, :, :, 1])  # 绘制第一个图(原始的图)的第二个特征图
# plt.imshow(output[1, :, :, 0])  # 绘制第二个图(原始的图)的第一个特征图
# plt.imshow(output[1, :, :, 1])  # 绘制第二个图(原始的图)的第二个特征图
plt.show()
#原始图片经过两个卷积核会形成两个featcure_map
#所以每一个图片会生成两个feature_map