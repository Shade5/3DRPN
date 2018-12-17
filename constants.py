# Training
data_path = '/home/a/workspace/katerina/3dmapping/double_tfrs_regress_small/'
Hdata = 128
Wdata = 128
BS = 2
lr = 5E-5
mom = 0.9
valp = 5

#General Parameters
HV = 18
VV = 3
MINH = 0
MAXH = 360
MINV = 1
MAXV = 31
HDELTA = (MAXH-MINH) / HV #20
VDELTA = (MAXV-MINV) / VV #10

N = 2


# Data Generation
min_objects = 1
max_objects = 3
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
