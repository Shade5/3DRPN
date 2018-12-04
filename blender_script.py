import bpy
import numpy as np
import pandas as pd
import sys
import pdb

def check_intersect(obj1, obj2):
    flag = True
    if abs(obj1.location[0]-obj2.location[0]) >= (obj1.dimensions[0] + obj2.dimensions[0])/2:
        flag = False

    if abs(obj1.location[1]-obj2.location[1]) >= (obj1.dimensions[2] + obj2.dimensions[2])/2:
        flag = False

    if flag:
        print("Intersect", obj1.name, obj2.name)
    return flag


def traslate_obj(translate_scale, scale_scale):
    x = bpy.context.scene.objects.active
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')

    initial_location = x.location.copy()
    initial_scale = x.scale.copy()

    while True:
        print("Testing:",  x.name)
        x.location = initial_location
        x.scale = initial_scale
        s = scale_scale*np.random.rand(1) + 2.4
        bpy.ops.transform.resize(value=(s, s, s))
        t = translate_scale*(np.random.rand(3, 1) - 0.5)
        r = np.pi * np.random.rand(1)
        t[2] = x.dimensions[1]/2
        bpy.ops.transform.translate(value=t)
        bpy.ops.transform.rotate(value=r, axis=(0, 0, 1))
        intersect = False
        for y in bpy.data.objects:
            if y != x and y.name != 'ground_plane':
                if check_intersect(x, y):
                    print("REPEAT")
                    intersect = True

        if not intersect:
            break

#for rotation
def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty

for i in range(100):
    print()

obj_paths = pd.read_csv('/home/neeraj/Documents/3D_PROJECT/3DRPN/paths.csv')
N = len(obj_paths['PATHS'])
path = '/home/neeraj/Documents/3D_PROJECT/3DRPN/mugs/'
#pdb.set_trace()
locs = []
dims = [] 
for i in range(5):
    r = np.random.randint(N)
    print("Adding:", obj_paths['PATHS'][r])
    bpy.ops.import_scene.obj(filepath=path + obj_paths['PATHS'][r] + '/models/model_normalized.obj')
    bpy.context.scene.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()
    bpy.context.scene.objects.active.name = obj_paths['PATHS'][r]
    traslate_obj(9 , 0.1)
    



bpy.ops.mesh.primitive_plane_add()
bpy.context.active_object.name = "ground_plane"
bpy.data.objects['ground_plane'].location = [0, 0, 0]
bpy.ops.transform.resize(value=(5, 5, 1))

scene = bpy.context.scene
cam = scene.objects['Camera']
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



lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')
lamp_data.energy = 1.0
# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)
# Link lamp object to the scene so it'll appear in this scene
scene.objects.link(lamp_object)
# Place lamp to a specified location
lamp_object.location = (-4.5,-1.69,4.86)

# And finally select it make active
lamp_object.select = True
#scene.objects.active = lamp_object

#for rendering
from math import radians

stepsize = 360.0 / 18
rotation_mode = 'XY'

fp = '/home/neeraj/Documents/3D_PROJECT/3DRPN/Images/'

for i in range(0, 18):
    z = 0
    for j in range(3):
        print("Height {},Rotation {}, {}".format((z+1)*1.5,(stepsize * i), radians(stepsize * i)))

        cam.location = (8,-8,(z+1)*1.5)

        bpy.data.scenes['Scene'].render.filepath = fp + '_rotation_{0:03d}_height_{1:01f}'.format(int(i * stepsize),(z+1)*1.5)
        z = (z + 1) % 3
        bpy.ops.render.render(write_still=True)  # render still
    b_empty.rotation_euler[2] += radians(stepsize)
