import bpy
import numpy as np
import pandas as pd
import sys
import os
import pdb
###add path of constants.py here
#sys.path.append('/home/neeraj/Documents/3D_PROJECT/3DRPN/')
import constants as const

def check_intersect(obj1, obj2):
	flag = True
	if abs(obj1.location[0]-obj2.location[0]) >= (obj1.dimensions[0] + obj2.dimensions[0])/2:
		flag = False

	if abs(obj1.location[1]-obj2.location[1]) >= (obj1.dimensions[2] + obj2.dimensions[2])/2:
		flag = False

	if flag:
		print("Intersect", obj1.name, obj2.name)
	return flag


def translate_obj(translate_scale, scale_scale):
	x = bpy.context.scene.objects.active
	bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')

	initial_location = x.location.copy()
	initial_scale = x.scale.copy()

	while True:
		print("Testing:",  x.name)
		x.location = initial_location
		x.scale = initial_scale
		s = scale_scale*np.random.rand(1) + 3.2
		bpy.ops.transform.resize(value=(s, s, s))
		t = translate_scale*(np.random.rand(3, 1) - 0.5)
		r = np.pi * np.random.rand(1)
		t[2] = x.dimensions[1]/2
		bpy.ops.transform.translate(value=t)
		bpy.ops.transform.rotate(value=r, axis=(0, 0, 1))
		intersect = False

		x_1, x_2 = x.location[0] + x.dimensions[0]/2, x.location[0] - x.dimensions[0]/2
		y_1, y_2 = x.location[1] + x.dimensions[2]/2, x.location[1] - x.dimensions[2]/2

		if x_1 > 5.0 or x_2 < -5.0:
			diff = (x_1 - 5.0) if x_1 > 5.0 else (x_2 - (-5))
			x.location[0] = x.location[0] - diff
		if y_1 > 5.0 or y_2 < -5.0:
			diff = (y_1 - 5.0) if y_1 > 5.0 else (y_2 - (-5))
			x.location[1] = x.location[1] - diff
		
		for y in bpy.data.objects:
			if y != x and y.name != 'ground_plane':
				if check_intersect(x, y):
					print("REPEAT")
					intersect = True

		if not intersect:
			break




def parent_obj_to_camera(b_camera):
	#for rotation
	origin = (0, 0, 0)
	b_empty = bpy.data.objects.new("Empty", None)
	b_empty.location = origin
	b_camera.parent = b_empty  # setup parenting

	scn = bpy.context.scene
	scn.objects.link(b_empty)
	scn.objects.active = b_empty
	return b_empty


##command line arguments

num_lamps = int(sys.argv[-5])
num_mugs = int(sys.argv[-4])
image_dir = sys.argv[-3]
save_file_name = sys.argv[-2]
voxel_file_name = sys.argv[-1]

#paths
obj_paths = pd.read_csv(const.obj_path_path)
N = len(obj_paths['PATHS'])
for i in range(num_mugs):
	r = np.random.randint(N)
	print("Adding:", obj_paths['PATHS'][r])
	bpy.ops.import_scene.obj(filepath=const.mugs_path + obj_paths['PATHS'][r] + '/models/model_normalized.obj')
	bpy.context.scene.objects.active = bpy.context.selected_objects[0]
	bpy.ops.object.join()
	bpy.context.scene.objects.active.name = obj_paths['PATHS'][r]
	translate_obj(9 , 0.1)

# Adding ground plane
bpy.ops.mesh.primitive_plane_add()
bpy.context.active_object.name = "ground_plane"
bpy.data.objects['ground_plane'].location = [0, 0, 0]
bpy.ops.transform.resize(value=(5, 5, 1))

#Rendering
scene = bpy.context.scene
render = scene.render
res_x = render.resolution_x = const.resolution
res_y  = render.resolution_y = const.resolution
scene.render.resolution_percentage = 100

ground_x, ground_y, _ = bpy.data.objects['ground_plane'].dimensions
add = ground_x/2
scale = res_x/ground_x

# Find bounding boxes and save
locs = []
dims = []
for ref in bpy.data.objects:
		if ref.name in ['Camera', 'Lamp', 'New Lamp', 'ground_plane']:
				pass
		else:
				locs.append([(ref.location.x + add)*scale, (ref.location.y + add)*scale, (ref.location.z + add)*scale])
				# Object is rotated along X axis by 90 by default, thus the dimensions along x and y are flipped
				dims.append([ref.dimensions.x*scale, ref.dimensions.z*scale, ref.dimensions.y*scale])

np.savez_compressed(save_file_name, locs=locs, dims=dims)


cam = scene.objects['Camera']
bpy.data.cameras['Camera'].lens = 22
cam.location = (8, -8, 0)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty


lamp = bpy.data.lamps['Lamp']
lamp.type = 'POINT'
lamp.energy = 2.0
lamp.shadow_method = 'RAY_SHADOW'
# Possibly disable specular shading:
lamp.use_specular = False

if num_lamps == 2:
	lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')
	lamp_data.energy = 1.0
	# Create new object with our lamp datablock
	lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)
	# Link lamp object to the scene so it'll appear in this scene
	scene.objects.link(lamp_object)
	# Place lamp to a specified location
	lamp_object.location = (-4.5, -1.69, 4.86)
	# Add one two more locations, the other edges of the square
	# And finally select it make active
	lamp_object.select = True



import math
from math import radians
HV = 18
VV = 3
MINH = 0
MAXH = 360 
MINV = 1
MAXV = 31 
HDELTA = (MAXH-MINH) / HV #20
VDELTA = (MAXV-MINV) / VV #10



def obj_centered_camera_pos(dist, azimuth_deg, elevation_deg):
	phi = float(elevation_deg) / 180 * math.pi
	theta = float(azimuth_deg) / 180 * math.pi
	x = (dist * math.cos(theta) * math.cos(phi))
	y = (dist * math.sin(theta) * math.cos(phi))
	z = (dist * math.sin(phi))
	return (x, y, z)

stepsize = HDELTA
rotation_mode = 'XY'


radius = np.random.choice([8,9,10,11])
camera_azimuth_angle = MINH
for i in range(HV):
	 
	camera_elevation_angle = MINV
	for j in range(VV):
		
		x,y,z = obj_centered_camera_pos(radius, camera_azimuth_angle, camera_elevation_angle)
		print("Height angle {},Rotation angle{}".format(j*VDELTA,i*HDELTA))
		cam.location = (x,y,z)
		bpy.data.scenes['Scene'].render.filepath = image_dir + '/_rotation_{0:03f}_height_{1:01f}'.format(i*HDELTA,j*VDELTA)
		
		bpy.ops.render.render(write_still=True)  # render still
		camera_elevation_angle += VDELTA
	camera_azimuth_angle += HDELTA
	b_empty.rotation_euler[2] += radians(stepsize)

# generating voxel occupancy
filepath = voxel_file_name + '.obj'
bpy.ops.export_scene.obj(filepath=filepath)
command = '%sutil/./binvox -d 128 %s'%(const.cwd, filepath)
os.system(command)