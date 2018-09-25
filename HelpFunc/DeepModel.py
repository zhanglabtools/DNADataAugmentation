# last time modified: 2018/9/25
# 模型加深

import tensorflow as tf
import tensorflow.contrib.slim as slim


class DeepModel(object):
    def __init__(self, channel=128):
        self.channel = channel

    def tostring(self):
        return str(self.channel)

    def build_net(self, x, keep_prob):
        # x需要是4×101×1的输入,也不一定,[None, 4, None, 1]应该就可以了,不过速度应该会减慢一点吧
        net = tf.pad(x, paddings=tf.constant([[0, 0], [0, 0], [12, 11], [0, 0]]))
        net = slim.conv2d(net,
                          num_outputs=self.channel,
                          kernel_size=[4, 24],
                          activation_fn=None,
                          padding="VALID")
        net = slim.batch_norm(net, activation_fn=slim.nn.relu)
        net = slim.conv2d(net, self.channel, [1, 12], activation_fn=None, padding="SAME")
        net = slim.batch_norm(net, activation_fn=slim.nn.relu)
        net = tf.reduce_max(net, axis=2, keepdims=True)
        net = slim.flatten(net)
        net = slim.nn.dropout(net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 2, activation_fn=slim.nn.softmax)
        return net