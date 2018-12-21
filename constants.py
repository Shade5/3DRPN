# Training
data_path = '/home/a/workspace/katerina/3dmapping/double_tfrs_regress_small/'
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
MINV = 1
MAXV = 31
HDELTA = (MAXH-MINH) / HV
VDELTA = (MAXV-MINV) / VV


# Data Generation
min_objects = 2
max_objects = 4
resolution = 128
obj_path_path = '/home/a/workspace/katerina/3DRPN/mugs/paths.csv'
mugs_path = '/home/a/workspace/katerina/3DRPN/mugs/'
BLENDER_DIR = '/home/a/workspace/katerina/blender/'
DATA_DIR = './DATA/'
cwd = './'
TF_RECORD_DIR = './data_tfrecords/'
anchor_size = 6
scale_factor = 4
default_z = 16
