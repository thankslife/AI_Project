import tensorflow as tf

# tf.Variable生成的变量，每次迭代都会变化，
# 这个变量也就是我们要去计算的结果，所以说你要计算什么，你是不是就把什么定义为Variable
x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2

# 创建一个计算图的一个上下文环境
sess = tf.Session()
# 碰到session.run()就会立刻去调用计算
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()


