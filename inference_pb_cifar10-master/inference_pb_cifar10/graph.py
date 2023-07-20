import tensorflow as tf
slim = tf.contrib.slim
import readcifar10
import os
import numpy as np


def model_fn_v1(net,keep_prob=0.5, is_training = True):

    batch_norm_params = {
    'is_training': is_training,
    'decay': 0.997,
    'epsilon': 1e-5,
    'scale': True,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    endpoints = {}

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0001),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:

                net = slim.conv2d(net, 32, [3, 3], activation_fn=None, normalizer_fn=None, scope='conv1')
                net = slim.conv2d(net, 32, [3, 3], activation_fn=None, normalizer_fn=None, scope='conv2')
                endpoints["conv2"] = net
                net = slim.max_pool2d(net, [3, 3], stride=2, scope="pool1")

                net = slim.conv2d(net, 64, [3, 3], activation_fn=None, normalizer_fn=None, scope='conv3')
                net = slim.conv2d(net, 64, [3, 3], activation_fn=None, normalizer_fn=None, scope='conv4')
                endpoints["conv4"] = net
                net = slim.max_pool2d(net, [3, 3], stride=2, scope="pool2")

                net = slim.conv2d(net, 128, [3, 3], activation_fn=None, normalizer_fn=None, scope='conv5')
                net = slim.conv2d(net, 128, [3, 3], activation_fn=None, normalizer_fn=None, scope='conv6')

                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                net = slim.flatten(net)
                net = slim.dropout(net, keep_prob, scope='dropout1')

                net = slim.fully_connected(net, 10, activation_fn=None, scope='fc2')
                endpoints["fc"] = net

    return net

def resnet_blockneck(net, kernel_size, down, stride, is_training):

    batch_norm_params = {
    'is_training': is_training,
    'decay': 0.997,
    'epsilon': 1e-5,
    'scale': True,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    shortcut = net

    with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(0.0001),
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME') as arg_sc:

                if kernel_size != net.get_shape().as_list()[-1]:
                    shortcut = slim.conv2d(net, kernel_size, [1, 1])

                if stride != 1:
                    shortcut = slim.max_pool2d(shortcut, [3, 3], stride=stride, scope="pool1")

                net = slim.conv2d(net, kernel_size // down, [1, 1])
                net = slim.conv2d(net, kernel_size // down, [3, 3])

                if stride != 1:
                    net = slim.max_pool2d(net, [3, 3], stride=stride, scope="pool1")

                net = slim.conv2d(net, kernel_size, [1, 1])

    net =  net + shortcut



    return net



def model_fn_resnet(net, keep_prob=0.5, is_training = True):

    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME') as arg_sc:

        net = slim.conv2d(net, 64, [3, 3], activation_fn=tf.nn.relu)
        net = slim.conv2d(net, 64, [3, 3], activation_fn=tf.nn.relu)

        net = resnet_blockneck(net, 128, 4, 2, is_training)
        net = resnet_blockneck(net, 128, 4, 1, is_training)
        net = resnet_blockneck(net, 256, 4, 2, is_training)
        net = resnet_blockneck(net, 256, 4, 1, is_training)
        net = resnet_blockneck(net, 512, 4, 2, is_training)
        net = resnet_blockneck(net, 512, 4, 1, is_training)

        #net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
        net = slim.flatten(net)

        net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc1')
        net = slim.dropout(net, keep_prob, scope='dropout1')
        net = slim.fully_connected(net, 10, activation_fn=None, scope='fc2')

    return net


def model(image, keep_prob=0.5, is_training=True):
    batch_norm_params = {
        "is_training": is_training,
        "epsilon": 1e-5,
        "decay": 0.997,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(0.0001),
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        net = slim.conv2d(image, 32, [3, 3], scope='conv1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv3')
        net = slim.conv2d(net, 64, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
        net = slim.conv2d(net, 128, [3, 3], scope='conv5')
        net = slim.conv2d(net, 128, [3, 3], scope='conv6')
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3')
        net = slim.conv2d(net, 256, [3, 3], scope='conv7')
        net = tf.reduce_mean(net, axis=[1, 2])  # nhwc--n11c
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024)
        net = slim.dropout(net, keep_prob)
        net = slim.fully_connected(net, 10)

    return net  # 10 dim vec


def func_optimal(loss_val):
    with tf.variable_scope("optimizer"):
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(0.0001, global_step,
                                                   decay_steps=10000,
                                                   decay_rate=0.99,
                                                   staircase=True)
        # ##更新 BN
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss_val, global_step)

        return optimizer, global_step, lr


def loss(logist, label):
    one_hot_label = slim.one_hot_encoding(label, 10)
    slim.losses.softmax_cross_entropy(logist, one_hot_label)

    reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    l2_loss = tf.add_n(reg_set)
    slim.losses.add_loss(l2_loss)

    totalloss = slim.losses.get_total_loss()

    return totalloss, l2_loss

def train_net():
    batchsize = 1
    model_path = "model"
    no_data = 0
 

    #images_train, labels_train = readcifar10.read(batchsize, 0, no_data)
 
    input_data  = tf.placeholder(tf.float32, shape=[1, 32, 32, 3], name="input_32")
 
    is_training = tf.placeholder(tf.bool, shape=1, name = "is_training")
    keep_prob   = tf.placeholder(tf.float32, shape=1, name= "keep_prob")

    logits      = model(input_data, keep_prob=1.0, is_training=False)
    softmax     = tf.nn.softmax(logits)
    pred_max        = tf.argmax(softmax, 1)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        ckpt = tf.train.latest_checkpoint(model_path)
        #
        #
        if ckpt:
            print("Model restored...",ckpt)
            saver.restore(sess, ckpt)

        print(pred_max)
        print(input_data)
        img_data = np.zeros([1, 32, 32, 3])
        sess.run(pred_max,{input_data: img_data})

        ##此处为核心内容，需要注意的
        output_graph_def = tf.graph_util. \
                        convert_variables_to_constants(sess,
                                                       sess.graph.as_graph_def(),
                                                       ['ArgMax'])

        with tf.gfile.FastGFile('output_graph.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    print("begin..")
    train_net()
