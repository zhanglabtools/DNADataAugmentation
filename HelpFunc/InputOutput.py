# last time modified: 2018/9/25
# 有关I/O的一些函数
# 一部分是直接从CSV变成想要的数据
# 另一部分是tensorflow接口的包装

import numpy as np
import pandas as pd
import tensorflow as tf
from HelpFunc.DNA import dna_one_hot_coding
from HelpFunc.MISC import get_now_time


# 0) 通用数据接口，给定对应的sequences和labels,就自动生成所需要的数据，而且batch size可以调节
# 当batch size等于1的时候，不进行sample维度的合并
def seq_label_to_nparray(seqs, labels, batchsize=512):
    features = [dna_one_hot_coding(i) for i in seqs]
    labels = [[1, 0] if i == 1 else [0, 1] for i in labels]
    if batchsize <= 1:
        return features, labels
    features = [np.stack(features[i:i + batchsize], 0) for i in range(0, len(features), batchsize)]
    labels = [np.stack(labels[i:i + batchsize], 0) for i in range(0, len(labels), batchsize)]
    return list(zip(features, labels))


# 1) 给定一个list的features和一个list的labels,将其存成tfrecords
# 通用的都用np.float32
def feature_label_to_tfrecords(features, labels, output_file):
    writer = tf.python_io.TFRecordWriter(path=output_file)
    for i in range(len(features)):
        feature_raw = np.array(features[i], np.float32).tostring()
        label_raw = np.array(labels[i], np.float32).tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "feature": tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature_raw])),
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw]))
                }
            )
        )
        writer.write(record=example.SerializeToString())
    writer.close()
    print("Finish writing to tfrecords: %r" % output_file)


# 2) DNA序列数据的解析器
def dna_parser(record, seqleng=101):
    features = tf.parse_single_example(record,
                                       features={
                                           'feature': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       })  # return one hot coding of DNA and label
    feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32), [4, seqleng, 1])
    label = tf.decode_raw(features['label'], tf.float32)
    return feature, label


# 3)另一个通用接口，一次性读取所有数据进内存
def read_all_data(filenames, parser, sess, num_parallel_calls=6, batchsize=128):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser, num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batchsize)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # 把所有数据存进来
    print(get_now_time(), "Starting reading data", filenames)
    data = []
    while True:
        try:
            ds = sess.run(next_element)
            data.append(ds)
        except tf.errors.OutOfRangeError:
            print(get_now_time(), "Finish reading data", filenames)
            break
    return data


# 4)专用函数，从csv到数据,只在这个项目里用
def csv_to_data(filename, batchsize=512, shuffle=False):
    print(get_now_time(), "Start loading data")
    temp = pd.read_csv(filename, header=None)
    if shuffle:
        temp = temp.sample(frac=1)
    temp.columns = ["location", "seqs", "labels"]
    labels = list(temp.pop("labels"))
    seqs = list(temp.pop("seqs"))
    test_data = seq_label_to_nparray(seqs, labels, batchsize=batchsize)
    print(get_now_time(), "Finish loading data")
    return test_data
