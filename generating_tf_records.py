import sys
import constants as const
from scipy.misc import imread
import numpy as np
import tensorflow as tf
import os
import re
from multiprocessing import Pool
from utils.tfutil import _bytes_feature
import glob
from box_overlaps import bbox_overlaps
from matplotlib import pyplot as plt
import tensorflow as tf
from munch import Munch
from os import path
import pdb
import shutil


def generating_probs_maps(anchor_size, boxes, feature_map_shape, scale_factor)
    anchor_size_pad = int(anchor_size / 2)
    objboxes = corner_to_standup_box2d(boxes, scale_factor)
    anchors = create_anchors(anchor_size, feature_map_shape)

    # draw_bbox2D(data[4], objboxes)

    pos_equal_one = np.zeros(feature_map_shape)
    neg_equal_one = np.zeros(feature_map_shape)

    iou = bbox_overlaps(np.ascontiguousarray(anchors).astype(np.float32),
                        np.ascontiguousarray(objboxes).astype(np.float32))

    # find anchor with highest iou(iou should also > 0)
    id_highest = np.argmax(iou.T, axis=1)
    id_highest_gt = np.arange(iou.T.shape[0])
    mask = iou.T[id_highest_gt, id_highest] > 0
    id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

    # find anchor iou > cfg.XXX_POS_IOU
    id_pos, id_pos_gt = np.where(iou > 0.6)

    # find anchor iou < cfg.XXX_NEG_IOU
    id_neg = np.where(np.sum(iou < 0.4, axis=1) == iou.shape[1])[0]

    id_pos = np.concatenate([id_pos, id_highest])
    id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

    id_pos, index = np.unique(id_pos, return_index=True)
    id_pos_gt = id_pos_gt[index]
    id_neg.sort()

    # cal the target and set the equal one
    index_x, index_y = np.unravel_index(id_pos, feature_map_shape - anchor_size)
    pos_equal_one[index_x + anchor_size_pad, index_y + anchor_size_pad] = 1
    index_x, index_y = np.unravel_index(id_neg, feature_map_shape - anchor_size)
    neg_equal_one[index_x + anchor_size_pad, index_y + anchor_size_pad] = 1

    # add_real_boxes(pos_equal_one, objboxes)
    #
    # plt.imshow(pos_equal_one)
    # plt.show()
    #
    # plt.imshow(neg_equal_one)
    # plt.show()

    return pos_equal_one.astype(int), neg_equal_one.astype(int)

def corner_to_standup_box2d(boxes_corner, scale_factor):
    N = len(boxes_corner)
    boxes2d = np.zeros((N,4))
    for i in range(N):
        x, y, z, l, w, h = boxes_corner[i].reshape((6, 1))
        boxes2d[i, :] = (x / scale_factor - l / (scale_factor*2), y / scale_factor - w / (scale_factor*2), \
         x / scale_factor + l / (scale_factor*2), y / scale_factor + w / (scale_factor*2))

    return boxes2d

def create_anchors(anchor_size, feature_map_shape):
    l, w = feature_map_shape
    anchors = np.zeros(((l - anchor_size)*(w - anchor_size), 4))
    for i in range(l - anchor_size):
        for j in range(w - anchor_size):
            anchors[i*(l - anchor_size) + j, :] = [i, j, i + anchor_size, j + anchor_size]

    return anchors

def get_regression_deltas(pos_equal_one, bboxes, anchor_size, scale_reduction):
    
    h, w = pos_equal_one.shape
    anchor_reg = np.zeros((h, w, 6))

    for i in range(bboxes.shape[0]):
    	bboxes[i] = (bboxes[i]/scale_reduction).astype(int)

    X_Idx, Y_Idx = np.where(pos_equal_one==1)
    for i in range(len(X_Idx)):
    	distances = []
    	for j in range(bboxes.shape[0]):
    		dist = np.linalg.norm(np.array(bboxes[i][0], bboxes[i][1]) - np.array(X_Idx[i], Y_Idx[i]))
    		distances.append(dist)
    	min_idx = np.min(distances)
    	anchor_reg[X_Idx[i], Y_Idx[i], :] = np.array((bboxes[min_idx])) - np.array((X_Idx[i],  Y_Idx[i],_, anchor_size, anchor_size, anchor_size))
    
    return anchor_reg



def controller_for_one_file(file_name):
	anchor_size, scale_factor = _, _
	feature_map_shape = np.array((_, _))
	images  = []
	bbox_coordinates = []
	image_names = glob.glob(files[i] + '/*.png')
	for j in range(len(image_names)):
		img = plt.imread(image_names[j])
		img = img[:,:,:3]
		images.append(img)
	data = np.load(files[i] + '/bboordinates.npz')
	dims = data['dims']
	locs = data['locs']
	for j in range(dims.shape[0]):
		x,y,z = locs[j]
		l,w,h = dims[j]
		bbox_coordinates.append([x,y,z,l,w,h])
	bbox_coordinates = np.stack(bbox_coordinates)
	pos_equal_one, neg_equal_one = generating_probs_maps(anchor_size, bbox_coordinates,feature_map_shape, scale_factor)
	anchors_reg = get_regression_deltas(pos_equal_one, bbox_coordinates, anchor_size, scale_factor)


def generate_tf_records(files, dump_dir)
	file_names = [re.split('/',x)[0] for x in files]
	
	for i in range(len(files)):
		images, bbox_coordinates, pos_equal_one, neg_equal_one, anchors_reg = controller_for_one_file(files[i])

		example = tf.train.Example(features=tf.train.Features(feature={
            'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images])),
            'bboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_coordinates])),
            'pos_equal_one': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pos_equal_one])),
            'neg_equal_one': tf.train.Feature(bytes_list=tf.train.BytesList(value=[neg_equal_one])),
            'anchor_reg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[anchor_reg.tostring()])),
        }))

        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        with tf.python_io.TFRecordWriter(dump_dir+str(i)+'.tfrecord', options=options) as writer:
            writer.write(example.SerializeToString())


		



if __name__ == "__main__":
	dump_dir = 'data_tfrecords/'
	if os.path.isdir(dump_dir):
		shutil.rmtree(dump_dir)
	os.mkdir(dump_dir)
	files = glob.glob('DATA/*')
	generate_tf_records(files, dump_dir)



