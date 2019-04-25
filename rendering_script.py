import bpy
import numpy as np
import pandas as pd
import sys
import os
import math
import glob
# add path of constants.py here
sys.path.append('/home/zhouxian/git/3DRPN/')
import constants as const
import pyquaternion


def check_intersect(obj1, obj2):
    flag = True
    if abs(obj1.location[0]-obj2.location[0]) >= (obj1.dimensions[0] + obj2.dimensions[0])/2:
        flag = False

    if abs(obj1.location[1]-obj2.location[1]) >= (obj1.dimensions[2] + obj2.dimensions[2])/2:
        flag = False

    if flag:
        print("Intersect", obj1.name, obj2.name)
    return flag


def translate_obj(translate_scale, scale_scale, scene_half_size, plane_height):
    x = bpy.context.scene.objects.active
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
    initial_location = x.location.copy()
    initial_scale = x.scale.copy()
    while True:
        print("Testing:",  x.name)
        x.location = initial_location
        x.scale = initial_scale
        s = scale_scale * np.random.uniform(low=0.6, high=1)
        bpy.ops.transform.resize(value=(s, s, s))
        t = translate_scale*(np.random.uniform(low=0, high=0.85, size=3) - 0.5) # 0.85 to reduce possibility of going out of scene
        r = np.pi * np.random.rand(1)
        t[2] = x.dimensions[1]/2 + plane_height
        bpy.ops.transform.translate(value=t)
        bpy.ops.transform.rotate(value=r, axis=(0, 0, 1))
        intersect = False
        x_1, x_2 = x.location[0] + x.dimensions[0]/2, x.location[0] - x.dimensions[0]/2
        y_1, y_2 = x.location[1] + x.dimensions[2]/2, x.location[1] - x.dimensions[2]/2

        if x_1 > scene_half_size or x_2 < -scene_half_size:
            diff = (x_1 - scene_half_size) if x_1 > scene_half_size else (x_2 - (-scene_half_size))
            x.location[0] = x.location[0] - diff
        if y_1 > scene_half_size or y_2 < -scene_half_size:
            diff = (y_1 - scene_half_size) if y_1 > scene_half_size else (y_2 - (-scene_half_size))
            x.location[1] = x.location[1] - diff
        for y in bpy.data.objects:
            if y != x and y.name != 'ground_plane':
                if check_intersect(x, y):
                    print("REPEAT")
                    intersect = True
        if not intersect:
            break

def parent_obj_to_camera(b_camera):
    # for rotation
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


def render_depth(yes):
    if yes:
        tree.links.new(tree.nodes["Render Layers"].outputs["Depth"], tree.nodes["Map Range"].inputs["Value"])
        tree.links.new(tree.nodes["Map Range"].outputs["Value"], tree.nodes["Composite"].inputs["Image"])

        # PNG as depth loses linearity, is not interpretable
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    else:
        tree.links.new(tree.nodes["Render Layers"].outputs["Image"], tree.nodes["Composite"].inputs["Image"])
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_depth = '8'

def obj_centered_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg + 90) / 180 * math.pi # 90deg correction to make view 0 is at the same position as ricson's data
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return x, y, z

def normalize_vec(vec):
    vec = np.array(vec)
    return vec/np.linalg.norm(vec)

def compute_camera_axes(cam_location, camera_azimuth_angle):
    ''' return camera axes follow opencv convention
    '''

    cam_z_axis = np.array([0, 0, 0]) - cam_location
    cam_x_axis = np.array([1.0*np.cos((180.0+camera_azimuth_angle)/180.0*np.pi),
                           1.0*np.sin((180.0+camera_azimuth_angle)/180.0*np.pi),
                           0])
    cam_y_axis = np.cross(cam_z_axis, cam_x_axis)
    cam_z_axis = normalize_vec(cam_z_axis)
    cam_x_axis = normalize_vec(cam_x_axis)
    cam_y_axis = normalize_vec(cam_y_axis)

    return cam_x_axis, cam_y_axis, cam_z_axis

def compute_extrinsic(x, y, z, R_cam):
    T_cam_to_blender_world = np.zeros((4, 4))
    T_cam_to_blender_world[:3, :3] = R_cam
    T_cam_to_blender_world[:, 3] = [x, y, z, 1.0]

    # now we get T w.r.t blender world frame, then we convert to the exported obj frame
    T_blender_to_export = np.array([
        [1.0, 0, 0, 0],
        [0, 0, 1.0, 0],
        [0, -1.0, 0, 0],
        [0, 0, 0, 1.0]])

    T_cam_to_world = T_blender_to_export.dot(T_cam_to_blender_world)

    return T_cam_to_world

def compute_intrinsic(img_size, fov_deg):
    f = img_size/2.0/np.tan(fov_deg/180.0/2.0*np.pi)
    K = np.array(
        [[f, 0, img_size/2.0],
        [0, f, img_size/2.0],
        [0, 0, 1]])
    return K

# command line arguments
split = sys.argv[-6]
num_lamps = int(sys.argv[-5])
num_mugs = int(sys.argv[-4])
image_dir = sys.argv[-3]
save_file_name = sys.argv[-2]
voxel_file_name = sys.argv[-1]

#paths
obj_paths = glob.glob(os.path.join(const.obj_folder_path, const.obj_cat, split, '*/models/model_normalized.obj'))

# whether to add ground plane
assert const.ground_plane in [True, False, 'rand']
if const.ground_plane == 'rand':
    add_ground_plane = np.random.choice([True, False])
else:
    add_ground_plane = const.ground_plane

# plane height randomization
if const.height_rand:
    plane_height = np.random.uniform(low=-0.7*const.scene_size/2.0,
                                     high=0.5*const.scene_size/2.0)
else:
    plane_height = 0

for i in range(num_mugs):
    obj_path = np.random.choice(obj_paths)
    obj_name = obj_path.split('/')[-3]
    print("Adding:", obj_name)
    bpy.ops.import_scene.obj(filepath=obj_path)
    bpy.context.scene.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.join()
    bpy.context.scene.objects.active.name = obj_name
    translate_obj(const.scene_size, const.obj_rescale, const.scene_size/2.0, plane_height)

if add_ground_plane:
    # Adding ground plane
    bpy.ops.mesh.primitive_plane_add()
    bpy.context.active_object.name = "ground_plane"
    bpy.data.objects['ground_plane'].location = [0, 0, plane_height]
    bpy.ops.transform.resize(value=(const.scene_size/2, const.scene_size/2, 1)) # original size is 2x2

# Rendering
scene = bpy.context.scene
render = scene.render
render.resolution_x = const.img_resolution
render.resolution_y = const.img_resolution
scene.render.resolution_percentage = 100


# set up camera
cam = scene.objects['Camera']
bpy.data.cameras['Camera'].lens_unit = 'FOV'
bpy.data.cameras['Camera'].angle = const.cam_FOV/180.0*np.pi
cam.location = (4, -4, 0)

# cam_constraint = cam.constraints.new(type='TRACK_TO')
# cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
# cam_constraint.up_axis = 'UP_Y'
# b_empty = parent_obj_to_camera(cam)
# cam_constraint.target = b_empty


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
    lamp_object.select = False
    # Link lamp object to the scene so it'll appear in this scene
    scene.objects.link(lamp_object)
    # Place lamp to a specified location
    lamp_object.location = (-4.5, -1.69, 4.86)
    # Add one two more locations, the other edges of the square
    # And finally select it make active
    lamp_object.select = True

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
tree.nodes.new('CompositorNodeMapRange')
# Range of depth (Hacky values set accordind to the location of the cameras for this project)
tree.nodes["Map Range"].inputs["From Min"].default_value = const.depth_render_min
tree.nodes["Map Range"].inputs["From Max"].default_value = const.depth_render_max


# generating voxel occupancy
i = 0
if add_ground_plane:
    bpy.data.objects['ground_plane'].select = False # individual voxel file does not contain ground plane

# !!! unselect all objects first before export obj files
for k, v in bpy.data.objects.items():
    if k not in ['Camera', 'Empty', 'ground_plane', 'Lamp', 'New Lamp']:
        v.select = False

for k, v in bpy.data.objects.items():
    if k not in ['Camera', 'Empty', 'ground_plane', 'Lamp', 'New Lamp']:
        v.select = True
        bpy.ops.export_scene.obj(filepath=voxel_file_name + str(i) + '.obj', use_selection=True)

        if const.generate_voxel:
            # command = '%sutil/./binvox -d 128 %s' % (const.cwd, voxel_file_name + str(i) + '.obj')
            command = '%sutil/binvox -aw -dc -down -pb -d %d -bb -1 -1 -1 1 1 1 -t binvox %s' % (const.cwd, const.vox_resolution*2, voxel_file_name + str(i) + '.obj') # resoluation 128
            os.system(command)
        v.select = False
        i += 1

filepath = voxel_file_name + '_all.obj'
bpy.ops.export_scene.obj(filepath=filepath)
if const.generate_voxel:
    # command = '%sutil/./binvox -d 128 %s' % (const.cwd, filepath)
    command = '%sutil/binvox -aw -dc -down -pb -d %d -bb -1 -1 -1 1 1 1 -t binvox %s' % (const.cwd, const.vox_resolution*2, filepath)
    os.system(command)

# Rendering Images
stepsize = const.HDELTA
cam.rotation_mode = 'QUATERNION'

radius = const.cam_dist
camera_elevation_angle = const.MINV
index = 0
T_cams = []
Ks = []
for i in range(const.VV):
    camera_azimuth_angle = const.MINH
    for j in range(const.HV):
        x, y, z = obj_centered_camera_pos(radius, camera_azimuth_angle, camera_elevation_angle)

        cam.location = (x, y, z)
        cam_x_axis, cam_y_axis, cam_z_axis = compute_camera_axes([x, y, z], camera_azimuth_angle)
        R_cam = np.vstack([cam_x_axis, cam_y_axis, cam_z_axis]).T
        # blender uses different camera frame convention
        R_cam_blender = np.vstack([cam_x_axis, -cam_y_axis, -cam_z_axis]).T
        cam_quat_blender = pyquaternion.Quaternion(matrix=R_cam_blender)
        cam.rotation_quaternion = [cam_quat_blender.w, cam_quat_blender.x, cam_quat_blender.y, cam_quat_blender.z]

        # extrinsics and intrinsics
        T_cams.append(compute_extrinsic(x, y, z, R_cam))
        Ks.append(compute_intrinsic(const.img_resolution, const.cam_FOV))

        # x, y, z = 0, 0, 4
        print("Height angle {},Rotation angle{}".format(i * const.VDELTA, j * const.HDELTA))
        render_depth(False)

        bpy.data.scenes['Scene'].render.filepath = image_dir + '/image_%d' % (index)
        bpy.ops.render.render(write_still=True)  # render still

        render_depth(True)
        bpy.data.scenes['Scene'].render.filepath = image_dir + '/depth_%d' % (index)

        bpy.ops.render.render(write_still=True)

        camera_azimuth_angle += const.HDELTA
        index += 1
    camera_elevation_angle += const.VDELTA

if const.store_matrices:
    Ks = np.stack(Ks)
    T_cams = np.stack(T_cams)
    np.save(os.path.join(image_dir, 'intrinsics.npy'), Ks)
    np.save(os.path.join(image_dir, 'extrinsics.npy'), T_cams)

