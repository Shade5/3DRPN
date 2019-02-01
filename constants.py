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
min_objects = 2
max_objects = 2
img_resolution = 128 # square image
vox_resolution = 128
obj_path_path = os.path.join(repo_path, 'mugs/paths.csv')
mugs_path = os.path.join(repo_path, 'mugs/')
DATA_DIR = os.path.join(repo_path, 'DATA/')
cwd = repo_path
TF_RECORD_DIR = os.path.join(repo_path, 'data_tfrecords/')
anchor_size = 6
scale_factor = 4
default_z = 16

num_files = 300

generate_voxel = True

cam_dist = 4 # radius of the sphere camera moves on
cam_FOV = 30 # horizontal FOV in deg
depth_render_min = 0 # 0.0 is distance 0
depth_render_max = 100 # 1.0 is distance 100
scene_size = 2
obj_rescale = 1.2