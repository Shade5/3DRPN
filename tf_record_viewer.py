import tensorflow as tf
import glob
import constants as const
import mayavi.mlab
import numpy as np
import pdb

def make_data(fns):
    data = tf.data.TFRecordDataset(fns, compression_type='GZIP')
    data = data.map(decode, num_parallel_calls=8)

    return data.make_one_shot_iterator()


def decode(example):
    stuff = tf.parse_single_example(example, features={
        'images': tf.FixedLenFeature([], tf.string),
        'bboxes': tf.FixedLenFeature([], tf.string),
        'pos_equal_one': tf.FixedLenFeature([], tf.string),
        'neg_equal_one': tf.FixedLenFeature([], tf.string),
        'anchor_reg': tf.FixedLenFeature([], tf.string),
        'num_obj': tf.FixedLenFeature([], tf.string),
        'voxel': tf.FixedLenFeature([], tf.string),
        'voxel_obj': tf.FixedLenFeature([], tf.string)
    })

    images = tf.decode_raw(stuff['images'], tf.float32)
    images = tf.reshape(images, (const.N, const.resolution, const.resolution, 3))
    bboxes = tf.decode_raw(stuff['bboxes'], tf.float64)
    bboxes = tf.reshape(bboxes, (-1, 6))
    pos_equal_one = tf.decode_raw(stuff['pos_equal_one'], tf.int64)
    pos_equal_one = tf.reshape(pos_equal_one, (32, 32))
    neg_equal_one = tf.decode_raw(stuff['neg_equal_one'], tf.int64)
    neg_equal_one = tf.reshape(neg_equal_one, (32, 32))
    anchor_reg = tf.decode_raw(stuff['anchor_reg'], tf.float64)
    anchor_reg = tf.reshape(anchor_reg, (32, 32, 6))
    num_obj = tf.decode_raw(stuff['num_obj'], tf.int64)
    voxel = tf.decode_raw(stuff['voxel'], tf.float32)
    voxel = tf.reshape(voxel, (128, 128, 128))
    voxel_obj = tf.decode_raw(stuff['voxel_obj'], tf.float64)
    voxel_obj = tf.reshape(voxel_obj, (const.max_objects, 128, 128, 128))
    return images, bboxes, pos_equal_one, neg_equal_one, anchor_reg, num_obj, voxel, voxel_obj


def draw_bbox_reg(center, dimension, A):
    A[center[0]:center[0]+dimension[0], center[1], center[2]] = 1
    A[center[0]:center[0] + dimension[0], center[1] + dimension[1], center[2]] = 1
    A[center[0], center[1]: center[1] + dimension[1], center[2]] = 1
    A[center[0] + dimension[0], center[1]: center[1] + dimension[1], center[2]] = 1

    A[center[0]:center[0]+dimension[0], center[1], center[2] + dimension[2]] = 1
    A[center[0]:center[0] + dimension[0], center[1] + dimension[1], center[2] + dimension[2]] = 1
    A[center[0], center[1]: center[1] + dimension[1], center[2] + dimension[2]] = 1
    A[center[0] + dimension[0], center[1]: center[1] + dimension[1], center[2] + dimension[2]] = 1

    A[center[0], center[1], center[2]:center[2] + dimension[2]] = 1
    A[center[0], center[1] + dimension[1], center[2]:center[2] + dimension[2]] = 1
    A[center[0] + dimension[0], center[1], center[2]:center[2] + dimension[2]] = 1
    A[center[0] + dimension[0], center[1] + dimension[1], center[2]:center[2] + dimension[2]] = 1


def anchors_viewer3D(pos_equal_one, anchor_reg, threshold=0.9, edgecolor=(0, 1, 0)):
    indices = np.array(np.where(pos_equal_one > threshold)).T
    A = np.zeros((const.resolution, const.resolution, const.resolution))
    for i, j in indices:
        X, Y, Z, L, W, H = [int(j) for j in const.scale_factor*(np.array([i, j, const.default_z, const.anchor_size, const.anchor_size, const.anchor_size]) + anchor_reg[i, j])]
        draw_bbox_reg((X-L//2, Y-W//2, Z-H//2), (L, W, H), A)
    xx, yy, zz = np.where(A == 1)
    mayavi.mlab.points3d(xx, yy, zz,
                         mode="cube",
                         color=edgecolor,
                         scale_factor=1)


fns = sorted(glob.glob(const.TF_RECORD_DIR + '*.tfrecord'))
iterator = make_data(fns)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for f in fns:
        print(f)
        images, bboxes, pos_equal_one, neg_equal_one, anchor_reg, num_obj, voxel, voxel_obj = sess.run(iterator.get_next())
        print(num_obj)
        anchors_viewer3D(pos_equal_one, anchor_reg, threshold=0.9, edgecolor=(0, 1, 0))
        mayavi.mlab.show()
