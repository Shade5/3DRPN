# 3DRPN
This branch contains code for generating data for the training pipeline in 3dmapping repo.

## Data Generation
* Set correct paths in constants.py
* Set number of files in wrapper.py
* Run wrapper.py to get data in DATA/
* Run generating_tf_records.py to make tfrecords in data_tfrecords/
* Run tf_record_viewer.py to view bbox in 3D for the saved tfrecords



