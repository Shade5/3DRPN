import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import constants as const
import mayavi.mlab
import numpy as np
import cv2

def make_data(fns):
    data = tf.data.TFRecordDataset(fns, compression_type='GZIP')
    # data = data.map(decode, num_parallel_calls=8)
    data = data.map(decode)

    return data.make_one_shot_iterator()


def decode(example):
    if const.store_matrices:
        stuff = tf.parse_single_example(example, features={
            'images': tf.FixedLenFeature([], tf.string),
            'depths': tf.FixedLenFeature([], tf.string),
            'objs_mask': tf.FixedLenFeature([], tf.string),
            'intrinsics': tf.FixedLenFeature([], tf.string),
            'extrinsics': tf.FixedLenFeature([], tf.string),
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
        Ks = tf.decode_raw(stuff['intrinsics'], tf.float64)
        Ks = tf.reshape(Ks, (const.N, 3, 3))
        Ts_cam_to_world = tf.decode_raw(stuff['extrinsics'], tf.float64)
        Ts_cam_to_world = tf.reshape(Ts_cam_to_world, (const.N, 4, 4))

        objs_mask = tf.decode_raw(stuff['objs_mask'], tf.float64)

        voxel = tf.decode_raw(stuff['voxel'], tf.float32)
        voxel = tf.reshape(voxel, (128, 128, 128))
        voxel_obj = tf.decode_raw(stuff['voxel_obj'], tf.float32)
        voxel_obj = tf.reshape(voxel_obj, (const.max_objects, 128, 128, 128))
        # return images, depths, bboxes, pos_equal_one, neg_equal_one, anchor_reg, num_obj, voxel, voxel_obj
        return images, depths, voxel, voxel_obj, Ks, Ts_cam_to_world, objs_mask

    else:
        stuff = tf.parse_single_example(example, features={
            'images': tf.FixedLenFeature([], tf.string),
            'depths': tf.FixedLenFeature([], tf.string),
            'objs_mask': tf.FixedLenFeature([], tf.string),
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
        objs_mask = tf.decode_raw(stuff['objs_mask'], tf.float64)
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
        return images, depths, voxel, voxel_obj, objs_mask


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
    A = np.zeros((const.img_resolution, const.img_resolution, const.img_resolution))
    for i, j in indices:
        X, Y, Z, L, W, H = [int(j) for j in const.scale_factor*(np.array([i, j, const.default_z, const.anchor_size, const.anchor_size, const.anchor_size]) + anchor_reg[i, j])]
        draw_bbox_reg((X-L//2, Y-W//2, Z-H//2), (L, W, H), A)
    xx, yy, zz = np.where(A == 1)
    mayavi.mlab.points3d(xx, yy, zz,
                         mode="cube",
                         color=edgecolor,
                         scale_factor=1)


def image_viewer(images, depths):
    for i in range(images.shape[0]):
        cv2.imshow("image", images[i])
        cv2.imshow("depth", depths[i])
        cv2.waitKey(100)

    cv2.destroyAllWindows()


fns = sorted(glob.glob(const.tf_record_dir + '*'))
iterator = make_data(fns)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for f in fns:
        
        print("File Name", f)
        # images, depths, bboxes, pos_equal_one, neg_equal_one, anchor_reg, num_obj, voxel, voxel_obj = sess.run(iterator.get_next())
        if const.store_matrices:
            images, depths, voxel, voxel_obj, Ks, Ts_cam_to_world, objs_mask = sess.run(iterator.get_next())
        else:
            images, depths, voxel, voxel_obj, objs_mask = sess.run(iterator.get_next())
        import IPython;IPython.embed()

        plt.imshow(images[0]); plt.show()
        plt.imshow(depths[0].squeeze()); plt.show()
        xx,yy,zz = np.where(voxel == 1)
        # zzz, yyy, xxx = np.where(np.transpose(voxel, [2, 1, 0]) == 1)
        mayavi.mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(0, 1, 0),
                             scale_factor=1)
        mayavi.mlab.show()
        # image_viewer(images, depths)
        # print("Number of objects", num_obj)
        # anchors_viewer3D(pos_equal_one, anchor_reg, threshold=0.9, edgecolor=(0, 1, 0))
        # mayavi.mlab.show()
