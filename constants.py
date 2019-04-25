import os

repo_path = '/home/zhouxian/git/3DRPN/'
BLENDER_DIR = '/home/zhouxian/blender-2.79b-linux-glibc219-x86_64/'


# Training
data_path = os.path.join(repo_path, 'blender_dataset/')
Hdata = 128
Wdata = 128
BS = 2
lr = 5E-5
mom = 0.9
valp = 5

#General Parameters
# # 3D RPN
# N = 2
# HV = 2
# VV = 1
# MINH = -90
# MINV = 0
# HDELTA = 90
# VDELTA = 0

# 54 Views
N = 54
HV = 18
VV = 3
MINH = 0
MAXH = 360
MINV = 20
MAXV = 80
HDELTA = (MAXH-MINH) / HV
VDELTA = (MAXV-MINV) / VV


# Data Generation
min_objects = 1
max_objects = 4
img_resolution = 128 # square image
vox_resolution = 128

ground_plane = False # [True, False, 'rand']
height_rand = False
store_matrices = False # whether to store extrinsics and intrinsics
# dataset_name = 'real'
# dataset_name = 'blender_rand_big'
# dataset_name = 'blender_rand_huge'
# dataset_name = 'blender_iccv'
# dataset_name = 'blender_rand_again'
# dataset_name = 'diverge_test'
# dataset_name = 'blender_flex_cam'
dataset_name = 'blender_var_obj'

train_num_files = 200
test_num_files = 50

obj_raw_folder_path = os.path.join(repo_path, 'raw_objs/')
obj_folder_path = os.path.join(repo_path, 'objs/')
obj_cat = '*' # regex expression for object catogory
data_dir = os.path.join(repo_path, 'data/', dataset_name, 'data_raw/')
video_path = os.path.join(repo_path, 'data/', dataset_name, 'blender_data.avi')
tf_record_dir = os.path.join(repo_path, 'data/', dataset_name, 'data_tfrecords/')
data_processed_dir = os.path.join(repo_path, 'data/', dataset_name, 'data_processed/')
cwd = repo_path
anchor_size = 6
scale_factor = 4
default_z = 16

generate_voxel = True

cam_dist = 4 # radius of the sphere camera moves on
cam_FOV = 30 # horizontal FOV in deg
depth_render_min = 0 # 0.0 is distance 0
depth_render_max = 100 # 1.0 is distance 100
scene_size = 2
obj_rescale = 1.2
