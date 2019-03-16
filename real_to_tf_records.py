import numpy as np
import imageio
import tensorflow as tf
import os
import re
import glob
import pdb
from util.box_overlaps import bbox_overlaps
from matplotlib import pyplot as plt
import shutil
import constants as const
import sys
import cv2
sys.path.append('util/')
from bv import read_bv
import pickle

def controller_for_one_file(file_name):
    return images, depths, voxel_full, voxels_individual

def get_new_K(fov_h, fov_v, W, H):
    fx = W/2.0 / np.tan(fov_h/2.0/180.0*np.pi)
    fy = H/2.0 / np.tan(fov_v/2.0/180.0*np.pi)
    
    new_K = np.array([
                [fx, 0, W/2.0],
                [0, fy, H/2.0],
                [0, 0, 1]])

    return new_K

def remove_distortion(img_array, fov_h=30, fov_v=30, W=128, H=128):
    '''
    re-render images as if it's taken by a standard perspective camera in simulation
    '''
    orig_H, orig_W = img_array.shape[1:3]

    # from d435 camera info
    orig_K = np.array([
        [615.3900146484375, 0.0, 326.35467529296875],
        [0.0, 615.323974609375, 240.33250427246094],
        [0.0, 0.0, 1.0]])

    new_K = get_new_K(fov_h, fov_v, W, H)

    M = new_K.dot(np.linalg.inv(orig_K))[:2]

    new_dim = np.array(img_array.shape)
    new_dim[1] = H
    new_dim[2] = W

    new_img_array = np.zeros(new_dim, dtype=img_array.dtype)
    for i in range(len(img_array)):
        new_img_array[i] = cv2.warpAffine(img_array[i], M, (W, H), cv2.INTER_AREA)

    return new_img_array

def generate_tf_records(file, tfrs_dump_dir, processed_dump_dir):
    rgb_array = data_file[b'rgb']
    depth_array = data_file[b'depth']
    radius = data_file[b'radius']
    good_ids = data_file[b'good_ids']
    good_i_ids = data_file[b'good_i_ids']
    good_j_ids = data_file[b'good_j_ids']
    scenehalfsize = data_file[b'scenehalfsize']
    T_world_to_optical = data_file[b'T_world_to_optical']
    print('good_ids: ', good_ids)
    print('good_i_ids: ', good_i_ids)
    print('good_j_ids: ', good_j_ids)
    print('radius: ', radius)
    print('scenehalfsize: ', scenehalfsize)

    fov = 30
    W = H = 128

    rgb_array = remove_distortion(rgb_array, fov_h=fov, fov_v=fov, W=W, H=H)
    depth_array = remove_distortion(depth_array, fov_h=fov, fov_v=fov, W=W, H=H)

    new_K = get_new_K(fov_h=fov, fov_v=fov, W=W, H=H)
    data_processed = {
        'T_world_to_optical': T_world_to_optical,
        'scenehalfsize': scenehalfsize,
        'rgb_array': rgb_array,
        'K': new_K
    }
    pickle.dump(data_processed, open(os.path.join(processed_dump_dir, 'data_small.p'), 'wb'))


    images = rgb_array.astype(np.float32)/255.0
    depths = depth_array.astype(np.float32)/1000.0 * (1.0/scenehalfsize) # mm to m, then scale to same as in simulation
    depths = depths/2.0 # compensate for the mysterious *2 operation in 3dmapping training pipline

    # Fake voxels
    voxel_full = np.zeros((128, 128, 128), dtype=np.float32)
    voxels_individual = np.zeros((2, 128, 128, 128), dtype=np.float32)
    voxels_individual[1, 64:, 64:, 64:] = 1
    voxels_individual[0, :32, :32, :32] = 1

    num_obj = voxels_individual.shape[0]
    voxels_individual = np.append(voxels_individual, np.zeros((const.max_objects - num_obj, 128, 128, 128), dtype=np.float32), axis=0)

    example = tf.train.Example(features=tf.train.Features(feature={
        'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(images).tostring()])),# float64
        'depths': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(depths).tostring()])),# float64
        'voxel': tf.train.Feature(bytes_list=tf.train.BytesList(value=[voxel_full.tostring()])),# int64
        'voxel_obj': tf.train.Feature(bytes_list=tf.train.BytesList(value=[voxels_individual.tostring()])),# int64
    }))
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    i = 1
    with tf.python_io.TFRecordWriter(tfrs_dump_dir+str(i), options=options) as writer:
        writer.write(example.SerializeToString())

if __name__ == "__main__":
    tfrs_dump_dir = const.tf_record_dir
    if os.path.isdir(tfrs_dump_dir):
        shutil.rmtree(tfrs_dump_dir)
    os.mkdir(tfrs_dump_dir)

    processed_dump_dir = const.data_processed_dir
    if os.path.isdir(processed_dump_dir):
        shutil.rmtree(processed_dump_dir)
    os.mkdir(processed_dump_dir)

    data_file = pickle.load(open(os.path.join(const.data_dir, 'data_small.p'), 'rb'), encoding='bytes')

    generate_tf_records(data_file, tfrs_dump_dir, processed_dump_dir)




