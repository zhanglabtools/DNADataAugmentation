# last time modified: 2018/6/18
# 最简单的一个模型,然而十分有效

import tensorflow as tf
import tensorflow.contrib.slim as slim


class BaseModel(object):
    def __init__(self, channel):
        self.channel = channel

    def tostring(self):
        return str(self.channel)

    def build_net(self, x, keep_prob):
        # x需要是4×101×1的输入
        net = tf.pad(x, paddings=tf.constant([[0, 0], [0, 0], [12, 11], [0, 0]]))
        net = slim.conv2d(net,
                          num_outputs=self.channel,
                          kernel_size=[4, 24],
                          activation_fn=slim.nn.relu,
                          padding="VALID")
        net = slim.nn.max_pool(net,
                               [1, 1, 101, 1],
                               strides=[1, 1, 1, 1],
                               padding="VALID")
        net = slim.flatten(net)
        net = slim.nn.dropout(net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 2, activation_fn=slim.nn.softmax)
        return net

    def build_net_151bp(self, x, keep_prob):
        # x需要是4×151×1的输入
        net = tf.pad(x, paddings=tf.constant([[0, 0], [0, 0], [12, 11], [0, 0]]))
        net = slim.conv2d(net,
                          num_outputs=self.channel,
                          kernel_size=[4, 24],
                          activation_fn=slim.nn.relu,
                          padding="VALID")
        net = slim.nn.max_pool(net,
                               [1, 1, 151, 1],
                               strides=[1, 1, 1, 1],
                               padding="VALID")
        net = slim.flatten(net)
        net = slim.nn.dropout(net, keep_prob=keep_prob)
        net = slim.fully_connected(net, 2, activation_fn=slim.nn.softmax)
        return net

    def build_dropout(self, x, keep_prob1, keep_prob2):
        # x需要是4×101×1的输入
        net = tf.pad(x, paddings=tf.constant([[0, 0], [0, 0], [12, 11], [0, 0]]))
        net = slim.nn.dropout(net, keep_prob=keep_prob1)
        net = slim.conv2d(net,
                          num_outputs=self.channel,
                          kernel_size=[4, 24],
                          activation_fn=slim.nn.relu,
                          padding="VALID")
        net = slim.nn.max_pool(net,
                               [1, 1, 101, 1],
                               strides=[1, 1, 1, 1],
                               padding="VALID")
        net = slim.flatten(net)
        net = slim.nn.dropout(net, keep_prob=keep_prob2)
        net = slim.fully_connected(net, 2, activation_fn=slim.nn.softmax)
        return net
