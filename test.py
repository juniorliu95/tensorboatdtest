import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# Summary：所有需要在TensorBoard上展示的统计结果。
# tf.name_scope()：为Graph中的Tensor添加层级，TensorBoard会按照代码指定的层级进行展示，
# 初始状态下只绘制最高层级的效果，点击后可展开层级看到下一层的细节。
# tf.summary.scalar()：添加标量统计结果。
# tf.summary.image(name,tensor,max_outputs): max_outputs is max num of
# batch elements to generate image for. tensor is 4-demension
# tf.summary.histogram()：添加任意shape的Tensor，统计这个Tensor的取值分布。
# tf.summary.merge_all()：添加一个操作，代表执行所有summary操作，这样可以避免人工执行每一个summary op。
# tf.summary.FileWrite：用于将Summary写入磁盘，需要制定存储路径logdir，如果传递了Graph对象，
# 则在Graph Visualization会显示Tensor Shape Information。执行summary op后，
# 将返回结果传递给add_summary()方法即可。
max_steps = 1000
learning_rate = 0.001
dropout = 0.9
data_dir = 'F:/projects/PycharmProjects/tensorboatdtest/input_data'
log_dir = 'F:/projects/PycharmProjects/tensorboatdtest/logs'
mnist = input_data.read_data_sets(data_dir, one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
with tf.name_scope('input_reshape'):
    image_reshaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_reshaped_input, 10)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


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


def nn_layer(input_tensor, input_dim, out_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, out_dim])
            variable_summaries(weights)
        with tf.name_scope('bias'):
            biases = bias_variable([out_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('preactivate', preactivate)
        activations = act(preactivate, name='activation')  # act参数为使用的方法
        tf.summary.histogram('activations', activations)
        return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
test_writer = tf.summary.FileWriter(log_dir+'test/', sess.graph)
tf.global_variables_initializer().run()


def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

saver = tf.train.Saver()  # 创建保存器
for i in range(max_steps):
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('accuracy at step %s is: %s' % (i, acc))
    else:
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()  # 为了保存训练时间和内存占用
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                                  options=run_options, run_metadata=run_metadata)  # 汇总结果
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)  # 训练元信息
            train_writer.add_summary(summary, global_step=i)
            saver.save(sess, log_dir+"/model.ckpt", i)
            print('Adding run metadata for ', i)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()



