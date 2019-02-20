import os
import shutil
import numpy as np
import sys
import multiprocessing
from joblib import Parallel, delayed
#sys.path.append('/home/neeraj/Documents/3D_PROJECT/3DRPN/')
import constants as const

if os.path.isdir(const.data_dir):
	shutil.rmtree(const.data_dir)
os.makedirs(const.data_dir)
def world_gen(i):
	num_lamps = np.random.choice([1, 2], p=[0.4, 0.6])
	num_mugs = np.random.randint(const.min_objects, const.max_objects + 1)
	
	image_dir = os.path.join(const.data_dir, '%d' % i)
	if os.path.isdir(image_dir):
		shutil.rmtree(image_dir)
	os.mkdir(image_dir)
	voxel_file_name = image_dir + '/voxel'
	save_file_name = image_dir+'/bboordinates.npz'
	command = '%sblender create_data.blend --background  --python rendering_script.py %d %d %s %s %s'%(const.BLENDER_DIR, num_lamps, num_mugs, image_dir, save_file_name, voxel_file_name)
	os.system(command)


num_cores=multiprocessing.cpu_count()
Parallel(n_jobs=int(num_cores))(delayed(world_gen)(i) for i in range(const.num_files))
