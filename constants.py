# Training
data_path = '/home/a/workspace/katerina/3dmapping/double_tfrs_regress_small/'
Hdata = 128
Wdata = 128
BS = 2
lr = 5E-5
mom = 0.9
valp = 5

# Data Generation
resolution = 128
obj_path_path = '/home/a/workspace/katerina/3DRPN/mugs/paths.csv'
mugs_path = '/home/a/workspace/katerina/3DRPN/mugs/'
BLENDER_DIR = '/home/a/workspace/katerina/blender/'
DATA_DIR = '/home/a/workspace/katerina/3DRPN/DATA/'
cwd = '/home/a/workspace/katerina/3DRPN/'
TF_RECORD_DIR = '/home/a/workspace/katerina/3DRPN/data_tfrecords/'
anchor_size = 6
scale_factor = 4
default_z = 16
