from random import shuffle, seed
import glob
import numpy as np
import tensorflow as tf
import keras
import sys, json, os
from pprint import pprint
import donkeycar.train as train
import collections
import matplotlib.pyplot as plt

import donkeycar as dk


train_filename      = 'tub_double_line.tfrecords'  # address to save the TFRecords file

tub_list = ['/home/philglau/Documents/tensorflow_projects/d2/data/double_line/tub_double_line',
            '/home/philglau/Documents/tensorflow_projects/d2/data/double_line/tub_double_line_pt2']

tub_list = ['/home/philglau/Documents/tensorflow_projects/d2/data/small_data_set']

tub_string = ','.join(tub_list)

cfg = dk.load_config(config_path='/home/philglau/Documents/tensorflow_projects/donkeycar/d2/config.py')

#get all the JSON records for the tubs passed in.
ordered_path_to_jsons = dk.utils.gather_records(cfg=cfg,tub_names=tub_string)

gen_records = {}

train.collate_records(ordered_path_to_jsons, gen_records, opts={'categorical':False})


key_list = sorted(list(gen_records.keys()))
pprint(key_list[0:5])
seed(666)
shuffle(key_list)
pprint(key_list[0:5])


ordered_records_dict = collections.OrderedDict(sorted(gen_records.items()))

i = 0
for k,v in ordered_records_dict.items():
    print ('key',k)
    #pprint(v)
    if i > 5:
        exit()
    i += 1

n_records =len(ordered_path_to_jsons)

print('Records to load:',n_records)

count = 0

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
i = 0

for record_path in ordered_path_to_jsons:
    # print how many images are saved every 1000 images
    if not i % 1000:
        print ('Train data: {}/{}'.format(i, n_records))
        sys.stdout.flush()

    with open(record_path, 'r') as fp:
        json_data = json.load(fp)

    basepath = os.path.dirname(record_path)
    image_filename = json_data["cam/image_array"]
    image_path = os.path.join(basepath, image_filename)

    angle    = float(json_data['user/angle'])
    throttle = float(json_data["user/throttle"])

    label   = np.array([angle,throttle],dtype=np.float32)
    img_arr = dk.utils.load_scaled_image_arr(image_path, cfg)

    try:
        # see if an IMU was used during recording of traning data
        accl_x = float(json_data['imu/acl_x'])
        accl_y = float(json_data['imu/acl_y'])
        accl_z = float(json_data['imu/acl_z'])

        gyro_x = float(json_data['imu/gyr_x'])
        gyro_y = float(json_data['imu/gyr_y'])
        gyro_z = float(json_data['imu/gyr_z'])

        imu_data = np.array([accl_x, accl_y, accl_z, gyro_x, gyro_y, gyro_z],dtype=np.float32)
    except:
        imu_data = np.zeros(6,dtype=np.float32)

    # Create a feature
    feature = {'train/label': _float_feature(value=[angle,throttle]),
               'train/imu':   _float_feature(value=imu_data),
               'train/image': _bytes_feature(value=tf.compat.as_bytes(img_arr.tostring()))}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    pprint(example)
    exit()
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    i += 1

writer.close()
sys.stdout.flush()