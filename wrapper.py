import bpy
import sys
import os
import shutil
import numpy as np
import time

BLENDER_DIR = '/home/neeraj/Documents/BLENDER/'
DATA_DIR = '/home/neeraj/Documents/3D_PROJECT/3DRPN/DATA/'
cwd = '/home/neeraj/Documents/3D_PROJECT/3DRPN/'
num_files = int(sys.argv[1])

if os.path.isdir(DATA_DIR):
	shutil.rmtree(DATA_DIR)
os.mkdir(DATA_DIR)
for i in range(num_files):
	num_lamps = np.random.choice([1,2],p=[0.4,0.6])
	num_mugs = np.random.choice(10)
	rot_step_size = 30
	
	image_dir = cwd + 'DATA/' + '%d'%(i)
	if os.path.isdir(image_dir):
		shutil.rmtree(image_dir)
	os.mkdir(image_dir)
	save_file_name = image_dir+'/bboordinates.npz'
	command = '%sblender create_data.blend --background  --python rendering_script.py %d %d %d %s %s'%(BLENDER_DIR,num_lamps,num_mugs, rot_step_size,image_dir,save_file_name)
	
	print('\n')
	print(command)
	print('\n')
	os.system(command)
	
	time.sleep(5)
	



