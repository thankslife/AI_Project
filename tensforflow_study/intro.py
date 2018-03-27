#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: intro.py

import tensorflow as tf

__author__ = 'yasaka'

# 1，构建图阶段

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # 默认就是类型tf.float32
print(node1, node2)

# 2，运行图阶段

sess = tf.Session()
print(sess.run([node1, node2]))

# 进行加和操作

node3 = tf.add(node1, node2)
print(node3)
print(sess.run(node3))

# A placeholder is a promise to provide a value later.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # tf.add(a,b)
print(sess.run(adder_node, {a: 3.0, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# 可以复用算子
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# 写个公式让程序计算
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
linear_model = W*x + b

# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# 接着给定y来计算平方损失函数
y = tf.placeholder(dtype=tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# 重新对变量人为的赋值
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# 不过机器学习的过程是一个自动找到最优解得过程，而不是人为的一遍遍赋值去尝试
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# 重新设置初始化值
sess.run(init)
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print(sess.run([W, b]))
