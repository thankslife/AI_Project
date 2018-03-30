import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


max_steps = 1000#最大迭代次数
learning_rate = 0.001#学习率
dropout = 0.9# 保留的数据
data_dir = './MNIST_data_bak'
log_dir = './logs/mnist_with_summaries'

mnist = input_data.read_data_sets(data_dir, one_hot=True)#把y这一列变成one_hot编码
sess = tf.InteractiveSession()

with tf.name_scope('input'):#with块中名字才是最重要的 一个块
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('input_reshape'):
    # 784维度变形为图片保持到节点
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])#-1代维度表不管有多少个  1代表1个通道 28*28个
    tf.summary.image('input', image_shaped_input, 10)#当做一个图片存起来


# 定义神经网络的初始化方法
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#截断的正态分布 这里可以用he_initinelize
    return tf.Variable(initial)#创建一个变量


def bias_variable(shape):#截距
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 以下代码是关于画图的
# 定义Variable变量的数据汇总函数，我们计算出变量的mean、stddev、max、min
# 对这些标量数据使用tf.summary.scalar进行记录和汇总
# 使用tf.summary.histogram直接记录变量var的直方图数据
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# 设计一个MLP多层神经网络来训练数据
# 在每一层中都对模型数据进行汇总
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):#定义一个隐藏层 input_dim上一层  output_dim本层输出
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])#shape传进来是上一层输入，本层输出 如果是MLP，就是全连接可以知道参数个数
            variable_summaries(weights)#把权重的各个指标（方差，平均值）进行总结
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases#带到激活函数之前的公式
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')#运用激活函数 函数里面传函数 高阶函数
        tf.summary.histogram('activations', activations)
        return activations


# 我们使用刚刚定义的函数创建一层神经网络，输入维度是图片的尺寸784=28*28
# 输出的维度是隐藏节点数500，再创建一个Dropout层，并使用tf.summary.scalar记录keep_prob

hidden1 = nn_layer(x, 784, 500, 'layer1')#建立第一层 隐藏层

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)#应用drop_out函数 保留下来的数据

# 然后使用nn_layer定义神经网络输出层，其输入维度为上一层隐含节点数500，输出维度为类别数10
# 同时激活函数为全等映射identity，暂时不使用softmax
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)#建立第二层 输出层

# 使用tf.nn.softmax_cross_entropy_with_logits()对前面的输出层的结果进行Softmax
# 处理并计算交叉熵损失cross_entropy，计算平均的损失，使用tf.summary.scalar进行统计汇总
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)#输出层给的结果logits=y #每一行的y是有10个数预测10个值 然后利用这10个值做归一化 然后具备一个概率的含义 第二步计算交叉熵
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)#平均损失
tf.summary.scalar('cross_entropy', cross_entropy)


# 下面使用Adam优化器对损失进行优化，同时统计预测正确的样本数并计算正确率accuracy，汇总
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)#AdamOptimizer比SGD更好一些，下降速度更快，更容易计算局部最优解 ，当数据量大的时候不如SGD
    #learning_rate虽然是固定的，后面会自适应，根据上一次的结果 所以大数据量的话，不如定义好策略，这样省时间
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))#预测值最大的索引 和真实值的索引
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#true 1 false 0 reduce_mean 是一个比例得到的结果

tf.summary.scalar('accuracy', accuracy)

# 因为我们之前定义了太多的tf.summary汇总操作，逐一执行这些操作太麻烦，
# 使用tf.summary.merge_all()直接获取所有汇总操作，以便后面执行
merged = tf.summary.merge_all()
# 定义两个tf.summary.FileWriter文件记录器再不同的子目录，分别用来存储训练和测试的日志数据
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')
# 同时，将Session计算图sess.graph加入训练过程，这样再TensorBoard的GRAPHS窗口中就能展示
# 整个计算图的可视化效果，最后初始化全部变量
tf.global_variables_initializer().run()


# 定义feed_dict函数，如果是训练，需要设置dropout，如果是测试，keep_prob设置为1
def feed_dict(train):
    if train:#如果是训练的话需要Droupout 测试的时候不要Droupout
        xs, ys = mnist.train.next_batch(100)#每一次拿一批次数据去训练
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels#真正测试的话全部测试，不是拿一批次的数据了
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}


# 执行训练、测试、日志记录操作
# 创建模型的保存器
saver = tf.train.Saver()
for i in range(max_steps):#max_steps迭代次数
    if i % 10 == 0:#每执行10次的时候汇总一次信息 然后计算测试集的一次准确率 因为传的是Flase
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)#然后写出
        print('Accuracy at step %s: %s' % (i, acc))
    else:
        if i % 100 == 99:#如果到100次
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()#保存的是元数据信息
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))# summary写的是跑完之后的数据
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, 1)#写文件
            saver.save(sess, log_dir + 'model.ckpt', i)
            print('Adding run metadata for', i)
        else:#不是10次，也不是100次 ，说明其他批次，则训练数据
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))#训练
            train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()






