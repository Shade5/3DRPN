# 3DRPN
Region proposal network that proposes 3D bounding boxes using two views of
the world[at 0 and 90 degrees]. 

This repo also contains code to generate renders using ShapeNet data/objects. The script creates the 3D world with random number of objects at different scales and locations with different lighting conditions. Images are rendered by moving the camera around the entire scene by fixed increments. At each angle of rotation of the camera, images are rendered at 3 different elevations. 


## Data Generation
* Set correct paths in constants.py
* Set number of files in wrapper.py
* Run wrapper.py to get data in DATA/
* Run generating_tf_records.py to make tfrecords in data_tfrecords/
* Run tf_record_viewer.py to view bbox in 3D for the saved tfrecords


## Bugs

* Goes out of bounds sometimes

## ToDo:

* Test for bugs




