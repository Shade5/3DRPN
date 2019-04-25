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
sys.path.append('util/')
from bv import read_bv


def generating_probs_maps(anchor_size, boxes, feature_map_shape, scale_factor):
    anchor_size_pad = int(anchor_size / 2)
    objboxes = corner_to_standup_box2d(boxes, scale_factor)
    anchors = create_anchors(anchor_size, feature_map_shape)

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
    boxes2d = np.zeros((N, 4))
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


def get_regression_deltas(pos_equal_one, bboxes, anchor_size, scale):
    h, w = pos_equal_one.shape
    anchor_reg = np.zeros((h, w, 6))
    bboxes = bboxes/scale

    X_Idx, Y_Idx = np.where(pos_equal_one == 1)
    for i in range(len(X_Idx)):
        distances = []
        for j in range(bboxes.shape[0]):
            distances.append((bboxes[j][0] - X_Idx[i])**2 + (bboxes[j][1] - Y_Idx[i])**2)
        min_idx = np.argmin(distances)
        anchor_reg[X_Idx[i], Y_Idx[i], :] = bboxes[min_idx] - (X_Idx[i],  Y_Idx[i], const.default_z, anchor_size, anchor_size, anchor_size)
    
    return anchor_reg


def find_int(splits):
    for i in range(len(splits)):
        split = splits[i]
        try:
            split = int(split)
            return split
        except:
            pass


def get_int_png(f):
    splits = re.split('_|.png',f)
    return find_int(splits)

def get_int_exr(f):
    splits = re.split('_|.exr',f)
    return find_int(splits)


def controller_for_one_file(file_name):
    feature_map_shape = np.array((32, 32))
    images = []
    depths = []
    bbox_coordinates = []
    image_names = glob.glob(file_name + '/image_*.png')
    image_names = sorted(image_names,key = get_int_png)
    depth_names = glob.glob(file_name + '/depth_*.exr')
    depth_names = sorted(depth_names, key = get_int_exr)
    voxel_full = read_bv(file_name + '/voxel_all.binvox').astype(np.float32)
    voxel_names = glob.glob(file_name + '/*.binvox')
    voxel_names.remove(file_name + '/voxel_all.binvox')
    voxels_individual = []
    for i in range(len(voxel_names)):
        vox = read_bv(voxel_names[i]).astype(np.float32)
        voxels_individual.append(vox)
    voxels_individual = np.stack(voxels_individual)
    for j in range(len(image_names)):
        img = plt.imread(image_names[j])[:, :, :3]
        images.append(img.astype(np.float32))
    images = np.stack(images)
    for j in range(len(depth_names)):
        depth = np.array(imageio.imread(depth_names[j], format='EXR-FI'))[:,:,0]
        depth = depth * (const.depth_render_max - const.depth_render_min) + const.depth_render_min # now we have real depth
        depth = depth / 2.0 # compensate for the mysterious *2 operation in 3dmapping training pipline
        depths.append(depth.astype(np.float32))
    depths = np.stack(depths)

    # data = np.load(file_name + '/bboordinates.npz')
    # dims = data['dims']
    # locs = data['locs']
    # for j in range(dims.shape[0]):
    #     x, y, z = locs[j]
    #     l, w, h = dims[j]
    #     bbox_coordinates.append([x, y, z, l, w, h])
    # bbox_coordinates = np.stack(bbox_coordinates)
    # pos_equal_one, neg_equal_one = generating_probs_maps(const.anchor_size, bbox_coordinates, feature_map_shape, const.scale_factor)
    # anchors_reg = get_regression_deltas(pos_equal_one, bbox_coordinates, const.anchor_size, const.scale_factor)
    # return images, depths, bbox_coordinates, pos_equal_one, neg_equal_one, anchors_reg, voxel_full, voxels_individual
    if const.store_matrices:
        Ks = np.load(os.path.join(file_name, 'intrinsics.npy'))
        T_cams = np.load(os.path.join(file_name, 'extrinsics.npy'))
        return images, depths, voxel_full, voxels_individual, Ks, T_cams
    else:
        return images, depths, voxel_full, voxels_individual


def generate_tf_records(files, dump_dir):
    for i in range(len(files)):
        print(i)
        # images, depths, bboxes, pos_equal_one, neg_equal_one, anchor_reg, voxel_full, voxels_individual = controller_for_one_file(files[i])
        if const.store_matrices:
            images, depths, voxel_full, voxels_individual, Ks, T_cams = controller_for_one_file(files[i])
        else:
            images, depths, voxel_full, voxels_individual = controller_for_one_file(files[i])


        num_obj = voxels_individual.shape[0]
        voxels_individual = np.append(voxels_individual, np.zeros((const.max_objects - num_obj, 128, 128, 128), dtype=np.float32), axis=0)
        objs_mask = np.zeros(const.max_objects)
        objs_mask[:num_obj] = 1.0
            

        if const.store_matrices:
            example = tf.train.Example(features=tf.train.Features(feature={
                'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(images).tostring()])),# float32
                'depths': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(depths).tostring()])),# float32
                'objs_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(objs_mask).tostring()])),# float64
                'intrinsics': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(Ks).tostring()])),# float64
                'extrinsics': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(T_cams).tostring()])),# float64
                # 'bboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bboxes.tostring()])),# float64
                # 'pos_equal_one': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pos_equal_one.tostring()])),# int64
                # 'neg_equal_one': tf.train.Feature(bytes_list=tf.train.BytesList(value=[neg_equal_one.tostring()])),# int64
                # 'anchor_reg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[anchor_reg.tostring()])),# float64
                # 'num_obj': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array([num_obj], dtype=np.int64).tostring()])),# int64
                'voxel': tf.train.Feature(bytes_list=tf.train.BytesList(value=[voxel_full.tostring()])),# float32
                'voxel_obj': tf.train.Feature(bytes_list=tf.train.BytesList(value=[voxels_individual.tostring()])),# float32
            }))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(images).tostring()])),# float32
                'depths': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(depths).tostring()])),# float32
                'objs_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(objs_mask).tostring()])),# float64
                # 'bboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bboxes.tostring()])),# float64
                # 'pos_equal_one': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pos_equal_one.tostring()])),# int64
                # 'neg_equal_one': tf.train.Feature(bytes_list=tf.train.BytesList(value=[neg_equal_one.tostring()])),# int64
                # 'anchor_reg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[anchor_reg.tostring()])),# float64
                # 'num_obj': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array([num_obj], dtype=np.int64).tostring()])),# int64
                'voxel': tf.train.Feature(bytes_list=tf.train.BytesList(value=[voxel_full.tostring()])),# float32
                'voxel_obj': tf.train.Feature(bytes_list=tf.train.BytesList(value=[voxels_individual.tostring()])),# float32
            }))


        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        with tf.python_io.TFRecordWriter(os.path.join(dump_dir, os.path.basename(files[i])), options=options) as writer:
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    dump_dir = const.tf_record_dir
    if os.path.isdir(dump_dir):
        shutil.rmtree(dump_dir)
    os.mkdir(dump_dir)
    files = glob.glob(os.path.join(const.data_dir, '*'))
    generate_tf_records(files, dump_dir)




