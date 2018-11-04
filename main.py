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
    data = data.shuffle(256)
    data = data.repeat()
    data = data.batch(const.BS)
    data = data.prefetch(4)
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

    pos_equal_one = tf.cast(tf.decode_raw(stuff['pos_equal_one'], tf.int64), tf.float32)
    pos_equal_one = tf.reshape(pos_equal_one, (32, 32))

    neg_equal_one = tf.cast(tf.decode_raw(stuff['neg_equal_one'], tf.int64), tf.float32)
    neg_equal_one = tf.reshape(neg_equal_one, (32, 32))

    anchor_reg = tf.cast(tf.decode_raw(stuff['anchor_reg'], tf.int64), tf.float32)
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


def first_layers(data_0, data_90):
    # Image from 0 degree
    x_0 = tf.layers.conv2d(data_0, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    x_0 = tf.layers.conv2d(x_0, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    x_0 = tf.layers.max_pooling2d(x_0, pool_size=(2, 2), strides=(2, 2), padding='same')

    # x_0 = tf.layers.conv2d(x_0, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # x_0 = tf.layers.conv2d(x_0, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # x_0 = tf.layers.max_pooling2d(x_0, pool_size=(2, 2), strides=(2, 2), padding='same')

    # Image from 90 degree
    x_90 = tf.layers.conv2d(data_90, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    x_90 = tf.layers.conv2d(x_90, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    x_90 = tf.layers.max_pooling2d(x_90, pool_size=(2, 2), strides=(2, 2), padding='same')

    # x_90 = tf.layers.conv2d(x_90, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # x_90 = tf.layers.conv2d(x_90, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # x_90 = tf.layers.max_pooling2d(x_90, pool_size=(2, 2), strides=(2, 2), padding='same')

    FT = tf.tile(x_0[:, :, :, None, :], [1, 1, 1, 64, 1])
    FT = FT + tf.tile(x_90[:, :, None, :, :], [1, 1, 64, 1, 1])

    return FT


def rpn(fl_input):
    temp_conv = tf.layers.conv3d(fl_input, 128, 3, strides=(2, 1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv3d(temp_conv, 64, 3, strides=(1, 1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv3d(temp_conv, 64, 3, strides=(2, 1, 1), activation=tf.nn.relu, padding="same")

    temp_conv = tf.transpose(temp_conv, perm=[0, 2, 3, 4, 1])
    temp_conv = tf.reshape(temp_conv, [-1, temp_conv.shape[1], temp_conv.shape[2], (temp_conv.shape[3]*temp_conv.shape[4])])
    # rpn
    # block1:
    temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(2, 2), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    deconv1 = tf.layers.conv2d_transpose(temp_conv, 256, 3, strides=(1, 1), padding="same")
    # block2:
    temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(2, 2), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    deconv2 = tf.layers.conv2d_transpose(temp_conv, 256, 2, strides=(2, 2), padding="same")
    # block3:
    temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(2, 2), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    temp_conv = tf.layers.conv2d(temp_conv, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="same")
    deconv3 = tf.layers.conv2d_transpose(temp_conv, 256, 4, strides=(4, 4), padding="SAME")

    # final:
    temp_conv = tf.concat([deconv3, deconv2, deconv1], -1)
    # Probability score map, scale = [None, 200/100, 176/120, 2]
    p_map = tf.layers.conv2d(temp_conv, 1, 1, strides=(1, 1), activation=tf.nn.relu, padding="valid")
    # Regression(residual) map, scale = [None, 200/100, 176/120, 14]
    r_map = tf.layers.conv2d(temp_conv, 6, 1, strides=(1, 1), activation=tf.nn.relu, padding="valid")
    p_pos = tf.reshape(tf.sigmoid(p_map), (-1, 32, 32))

    return p_pos, r_map


def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
        tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1


def calc_loss(p_pos, r_map, pos_equal_one, anchors_reg):
    pos_equal_one_sum = tf.reduce_sum(pos_equal_one, axis=[1, 2])
    neg_equal_one_sum = tf.reduce_sum(1 - pos_equal_one, axis=[1, 2])
    cls_pos_loss = (-pos_equal_one * tf.log(p_pos + 1e-6)) / tf.reshape(pos_equal_one_sum, [-1, 1, 1])
    cls_neg_loss = (-(1 - pos_equal_one) * tf.log(1 - p_pos + 1e-6)) / tf.reshape(neg_equal_one_sum, [-1, 1, 1])
    loss_prob = tf.reduce_sum(cls_pos_loss + cls_neg_loss) / const.BS

    pos_equal_one_expanded = tf.expand_dims(pos_equal_one, 3)
    r_map_mask = tf.tile(pos_equal_one_expanded, [1, 1, 1, 6])
    loss_reg = tf.reduce_sum(smooth_l1(r_map * r_map_mask, anchors_reg * r_map_mask) / tf.reshape(pos_equal_one_sum, [-1, 1, 1, 1])) / const.BS

    loss = loss_prob + loss_reg

    return loss


iterator = make_data('double_train')

data = iterator.get_next()
FT = first_layers(data[0][:, 0], data[0][:, 5])
p_pos, r_map = rpn(FT)
loss = calc_loss(p_pos, r_map, data[4], data[6])
opt = tf.train.AdamOptimizer(const.lr, const.mom).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(300):
        sess.run(opt)



