import os
import shutil
import numpy as np
import sys
#sys.path.append('/home/neeraj/Documents/3D_PROJECT/3DRPN/')
import constants as const


num_files = 1

if os.path.isdir(const.DATA_DIR):
	shutil.rmtree(const.DATA_DIR)
os.mkdir(const.DATA_DIR)
for i in range(num_files):
	num_lamps = np.random.choice([1, 2], p=[0.4, 0.6])
	num_mugs = np.random.randint(2, 8)
	
	image_dir = const.cwd + 'DATA/' + '%d'%(i)
	if os.path.isdir(image_dir):
		shutil.rmtree(image_dir)
	os.mkdir(image_dir)
	voxel_file_name = image_dir + '/voxel'
	save_file_name = image_dir+'/bboordinates.npz'
	command = '%sblender create_data.blend --background  --python rendering_script_3drpn.py %d %d %s %s %s' % (const.BLENDER_DIR, num_lamps, num_mugs, image_dir, save_file_name, voxel_file_name)
	os.system(command)
	



