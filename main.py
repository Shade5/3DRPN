import tensorflow as tf
import constants as const
import numpy as np
import mayavi.mlab
from matplotlib import pyplot as plt


def make_data(fn):
    with open('lists/' + fn, 'r') as f:
        fns = f.readlines()
    fns = [const.data_path + fn.strip() + ".tfrecord" for fn in fns]
    data = tf.data.TFRecordDataset(fns, compression_type='GZIP')
    data = data.map(decode, num_parallel_calls=8)
    data = data.batch(const.BS)
    iterator = data.make_one_shot_iterator()

    return iterator


def decode(example):
    stuff = tf.parse_single_example(example, features={
        'images': tf.FixedLenFeature([], tf.string),
        'zmaps': tf.FixedLenFeature([], tf.string),
        'segs': tf.FixedLenFeature([], tf.string),
        'voxel': tf.FixedLenFeature([], tf.string),
        'obj1': tf.FixedLenFeature([], tf.string),
        'obj2': tf.FixedLenFeature([], tf.string),
        'bbox1': tf.FixedLenFeature([], tf.string),
        'bbox2': tf.FixedLenFeature([], tf.string),
        'pos_equal_one': tf.FixedLenFeature([], tf.string),
        'neg_equal_one': tf.FixedLenFeature([], tf.string),
        'anchor_reg': tf.FixedLenFeature([], tf.string),
    })

    N = 54
    images = tf.decode_raw(stuff['images'], tf.float32)
    images = tf.reshape(images, (N, const.Hdata, const.Wdata, 4))
    images = tf.slice(images, [0, 0, 0, 0], [-1, -1, -1, 3])

    voxel = tf.decode_raw(stuff['voxel'], tf.float32)
    voxel = tf.reshape(voxel, (128, 128, 128))

    bbox1 = tf.decode_raw(stuff['bbox1'], tf.int64)
    bbox1 = tf.reshape(bbox1, (2, 3))

    bbox2 = tf.decode_raw(stuff['bbox2'], tf.int64)
    bbox2 = tf.reshape(bbox2, (2, 3))

    pos_equal_one = tf.decode_raw(stuff['pos_equal_one'], tf.int64)
    pos_equal_one = tf.reshape(pos_equal_one, (32, 32))

    neg_equal_one = tf.decode_raw(stuff['neg_equal_one'], tf.int64)
    neg_equal_one = tf.reshape(neg_equal_one, (32, 32))

    anchor_reg = tf.decode_raw(stuff['anchor_reg'], tf.int64)
    anchor_reg = tf.reshape(anchor_reg, (32, 32, 6))

    return images, voxel, bbox1, bbox2, pos_equal_one, neg_equal_one, anchor_reg


def draw_bbox(voxels, bbox):
    voxels[(bbox[0][0] - int(bbox[1][0] / 2)):(bbox[0][0] + int(bbox[1][0] / 2)), (bbox[0][1] + int(bbox[1][1] / 2)), (bbox[0][2] + int(bbox[1][2] / 2))] = 1
    voxels[(bbox[0][0] - int(bbox[1][0] / 2)):(bbox[0][0] + int(bbox[1][0] / 2)), (bbox[0][1] - int(bbox[1][1] / 2)), (bbox[0][2] + int(bbox[1][2] / 2))] = 1
    voxels[(bbox[0][0] - int(bbox[1][0] / 2)):(bbox[0][0] + int(bbox[1][0] / 2)), (bbox[0][1] + int(bbox[1][1] / 2)), (bbox[0][2] - int(bbox[1][2] / 2))] = 1
    voxels[(bbox[0][0] - int(bbox[1][0] / 2)):(bbox[0][0] + int(bbox[1][0] / 2)), (bbox[0][1] - int(bbox[1][1] / 2)), (bbox[0][2] - int(bbox[1][2] / 2))] = 1

    voxels[(bbox[0][0] + int(bbox[1][0] / 2)), (bbox[0][1] - int(bbox[1][1] / 2)):(bbox[0][1] + int(bbox[1][1] / 2)), (bbox[0][2] + int(bbox[1][2] / 2))] = 1
    voxels[(bbox[0][0] - int(bbox[1][0] / 2)), (bbox[0][1] - int(bbox[1][1] / 2)):(bbox[0][1] + int(bbox[1][1] / 2)), (bbox[0][2] + int(bbox[1][2] / 2))] = 1
    voxels[(bbox[0][0] + int(bbox[1][0] / 2)), (bbox[0][1] - int(bbox[1][1] / 2)):(bbox[0][1] + int(bbox[1][1] / 2)), (bbox[0][2] - int(bbox[1][2] / 2))] = 1
    voxels[(bbox[0][0] - int(bbox[1][0] / 2)), (bbox[0][1] - int(bbox[1][1] / 2)):(bbox[0][1] + int(bbox[1][1] / 2)), (bbox[0][2] - int(bbox[1][2] / 2))] = 1

    voxels[(bbox[0][0] + int(bbox[1][0] / 2)), (bbox[0][1] + int(bbox[1][1] / 2)), (bbox[0][2] - int(bbox[1][2] / 2)):(bbox[0][2] + int(bbox[1][2] / 2))] = 1
    voxels[(bbox[0][0] - int(bbox[1][0] / 2)), (bbox[0][1] + int(bbox[1][1] / 2)), (bbox[0][2] - int(bbox[1][2] / 2)):(bbox[0][2] + int(bbox[1][2] / 2))] = 1
    voxels[(bbox[0][0] + int(bbox[1][0] / 2)), (bbox[0][1] - int(bbox[1][1] / 2)), (bbox[0][2] - int(bbox[1][2] / 2)):(bbox[0][2] + int(bbox[1][2] / 2))] = 1
    voxels[(bbox[0][0] - int(bbox[1][0] / 2)), (bbox[0][1] - int(bbox[1][1] / 2)), (bbox[0][2] - int(bbox[1][2] / 2)):(bbox[0][2] + int(bbox[1][2] / 2))] = 1


iterator = make_data('double_train')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(300):
        data = sess.run(iterator.get_next())

        # Image from 0 degree
        x_0 = tf.layers.conv2d(data[0][:, 0], filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        x_0 = tf.layers.conv2d(x_0, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        x_0 = tf.layers.max_pooling2d(x_0, pool_size=(2, 2), strides=(2, 2), padding='same')

        x_0 = tf.layers.conv2d(x_0, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        x_0 = tf.layers.conv2d(x_0, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        x_0 = tf.layers.max_pooling2d(x_0, pool_size=(2, 2), strides=(2, 2), padding='same')

        # Image from 90 degree
        x_90 = tf.layers.conv2d(data[0][:, 5], filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        x_90 = tf.layers.conv2d(x_90, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        x_90 = tf.layers.max_pooling2d(x_90, pool_size=(2, 2), strides=(2, 2), padding='same')

        x_90 = tf.layers.conv2d(x_90, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        x_90 = tf.layers.conv2d(x_90, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        x_90 = tf.layers.max_pooling2d(x_90, pool_size=(2, 2), strides=(2, 2), padding='same')

