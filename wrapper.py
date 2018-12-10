import os
import shutil
import numpy as np
import constants as const

num_files = 10


if os.path.isdir(const.DATA_DIR):
	shutil.rmtree(const.DATA_DIR)
os.mkdir(const.DATA_DIR)
for i in range(num_files):
	num_lamps = np.random.choice([1,2],p=[0.4,0.6])
	num_mugs = np.random.randint(2,8)
	
	image_dir = const.cwd + 'DATA/' + '%d'%(i)
	if os.path.isdir(image_dir):
		shutil.rmtree(image_dir)
	os.mkdir(image_dir)
	save_file_name = image_dir+'/bboordinates.npz'
	command = '%sblender create_data.blend --background  --python rendering_script.py %d %d %s %s'%(const.BLENDER_DIR,num_lamps,num_mugs,image_dir,save_file_name)
	os.system(command)



