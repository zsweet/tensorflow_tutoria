#placehoder
import  tensorflow as tf
import numpy as np

def add_layer (inputs,in_size,out_size,activation_function=None):#activation_function没有激活函数就相当于线性函数

    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # print(sess.run(inputs))

    Weithts = tf.Variable(tf.random_normal([in_size,out_size]))
    print('>>>>>',inputs,Weithts)
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)

    Wx_plus_b = tf.matmul(inputs,Weithts)+biases
    if activation_function is None :
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return  outputs

def neural_network():
    #生成数据
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    # 定义两层
    xs = tf.placeholder(tf.float32, [None, 1])  # None用来限制用例个数
    ys = tf.placeholder(tf.float32, [None, 1])
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    prediction = add_layer(l1, 10, 1, activation_function=None)

    #定义递归下降
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

if  __name__  == '__main__':
  neural_network()



