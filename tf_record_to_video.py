import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import constants as const
import mayavi.mlab
import numpy as np
import cv2
import os

def make_data(fns):
    data = tf.data.TFRecordDataset(fns, compression_type='GZIP')
    data = data.map(decode)
    return data.make_one_shot_iterator()

def check_dir(f):
    if not os.path.exists(os.path.dirname(f)):
        os.makedirs(os.path.dirname(f))

def decode(example):
    stuff = tf.parse_single_example(example, features={
        'images': tf.FixedLenFeature([], tf.string),
        'depths': tf.FixedLenFeature([], tf.string),
        # 'bboxes': tf.FixedLenFeature([], tf.string),
        # 'pos_equal_one': tf.FixedLenFeature([], tf.string),
        # 'neg_equal_one': tf.FixedLenFeature([], tf.string),
        # 'anchor_reg': tf.FixedLenFeature([], tf.string),
        # 'num_obj': tf.FixedLenFeature([], tf.string),
        'voxel': tf.FixedLenFeature([], tf.string),
        'voxel_obj': tf.FixedLenFeature([], tf.string)
    })

    images = tf.decode_raw(stuff['images'], tf.float32)
    images = tf.reshape(images, (const.N, const.img_resolution, const.img_resolution, 3))
    depths = tf.decode_raw(stuff['depths'], tf.float32)
    depths = tf.reshape(depths, (const.N, const.img_resolution, const.img_resolution, 1))
    # bboxes = tf.decode_raw(stuff['bboxes'], tf.float64)
    # bboxes = tf.reshape(bboxes, (-1, 6))
    # pos_equal_one = tf.decode_raw(stuff['pos_equal_one'], tf.int64)
    # pos_equal_one = tf.reshape(pos_equal_one, (32, 32))
    # neg_equal_one = tf.decode_raw(stuff['neg_equal_one'], tf.int64)
    # neg_equal_one = tf.reshape(neg_equal_one, (32, 32))
    # anchor_reg = tf.decode_raw(stuff['anchor_reg'], tf.float64)
    # anchor_reg = tf.reshape(anchor_reg, (32, 32, 6))
    # num_obj = tf.decode_raw(stuff['num_obj'], tf.int64)
    voxel = tf.decode_raw(stuff['voxel'], tf.float32)
    voxel = tf.reshape(voxel, (128, 128, 128))
    voxel_obj = tf.decode_raw(stuff['voxel_obj'], tf.float32)
    voxel_obj = tf.reshape(voxel_obj, (const.max_objects, 128, 128, 128))
    # return images, depths, bboxes, pos_equal_one, neg_equal_one, anchor_reg, num_obj, voxel, voxel_obj
    return images, depths, voxel, voxel_obj

check_dir(const.video_path)

out = cv2.VideoWriter(const.video_path, cv2.VideoWriter_fourcc(*'MJPG'), 5, (const.img_resolution, const.img_resolution))
fns = sorted(glob.glob(os.path.join(const.tf_record_dir, '*')))
iterator = make_data(fns)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for f in fns[:20]:
        print(f)
        images, depths, voxel, voxel_obj = sess.run(iterator.get_next())
        for image in images:
            out.write(np.array(image*255, dtype=np.uint8))

out.release()
