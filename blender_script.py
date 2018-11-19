import bpy
import numpy as np
import pandas as pd


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
        s = scale_scale*np.random.rand(1) + 13
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


for i in range(100):
    print()
obj_paths = pd.read_csv('/home/a/workspace/katerina/3DRPN/paths.csv')
N = len(obj_paths['PATHS'])
path = '/home/a/Downloads/mugs/'

for i in range(30):
    r = np.random.randint(N)
    print("Adding:", obj_paths['PATHS'][r])
    bpy.ops.import_scene.obj(filepath=path + obj_paths['PATHS'][r] + '/models/model_normalized.obj')
    bpy.context.scene.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()
    bpy.context.scene.objects.active.name = obj_paths['PATHS'][r]
    traslate_obj(128, 3)


bpy.ops.mesh.primitive_plane_add()
bpy.context.active_object.name = "ground_plane"
bpy.data.objects['ground_plane'].location = [0, 0, 0]
bpy.ops.transform.resize(value=(64, 64, 1))

