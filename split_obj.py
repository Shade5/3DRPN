import constants as const
import os
import glob
import shutil
import numpy as np

np.random.seed(0)

train_split = 0.8

raw_obj_dir = const.obj_raw_folder_path
obj_dir = const.obj_folder_path
if os.path.isdir(obj_dir):
    shutil.rmtree(obj_dir)
os.mkdir(obj_dir)
obj_cats = os.listdir(raw_obj_dir)

for obj_cat in obj_cats:
    obj_cat_dir = os.path.join(obj_dir, obj_cat)
    obj_cat_train_dir = os.path.join(obj_cat_dir, 'train')
    obj_cat_test_dir = os.path.join(obj_cat_dir, 'test')
    os.makedirs(obj_cat_train_dir)
    os.makedirs(obj_cat_test_dir)

    raw_obj_cat_dir = os.path.join(raw_obj_dir, obj_cat)
    obj_cat_ids = os.listdir(raw_obj_cat_dir)

    np.random.shuffle(obj_cat_ids)
    n_ids = len(obj_cat_ids)
    split_id = int(n_ids * train_split)

    obj_cat_ids_train = obj_cat_ids[:split_id]
    obj_cat_ids_test = obj_cat_ids[split_id:]

    for obj_cat_id in obj_cat_ids_train:
        raw_obj_cat_id_dir = os.path.join(raw_obj_cat_dir, obj_cat_id)
        print(raw_obj_cat_id_dir)
        command = 'cp -r %s %s'%(raw_obj_cat_id_dir, obj_cat_train_dir)
        os.system(command)

    for obj_cat_id in obj_cat_ids_test:
        raw_obj_cat_id_dir = os.path.join(raw_obj_cat_dir, obj_cat_id)
        print(raw_obj_cat_id_dir)
        command = 'cp -r %s %s'%(raw_obj_cat_id_dir, obj_cat_test_dir)
        os.system(command)
