from donkeycar.utils import *
import numpy as np
import json
import h5py
from pprint import pprint
from time import sleep
import matplotlib.pyplot as plt

import donkeycar as dk



tub_list = '/home/philglau/Documents/tensorflow_projects/d2/data/double_line/tub_double_line,' \
           +'/home/philglau/Documents/tensorflow_projects/d2/data/double_line/tub_double_line_pt2'

cfg = dk.load_config(config_path='/home/philglau/Documents/tensorflow_projects/donkeycar/d2/config.py')

#get all the JSON records for the tubs passed in.
ordered_path_to_jsons = gather_records(cfg=cfg,tub_names=tub_list)

n_records =len(ordered_path_to_jsons)

print('Records to load:',n_records)

count = 0
gen_records = []

hdf5_path           = '/home/philglau/Documents/tensorflow_projects/d2/data/double_set'+'.hdf5'  # address to where you want to save the hdf5 file
img_dtype           = np.uint8

# (n)umber of images by H,W,C
train_img_shape         = (n_records, cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
# imu has 6 float32s. optional. Zero if not present.
train_imu_shape         = (n_records,6)
# labels are [angle and throttle as scalar float32s
label_shape             = (n_records,2)

mean_img = np.zeros(train_img_shape[1:], np.float32)

hdf5_file = h5py.File(hdf5_path, mode='w')

# create an structure for the training images
hdf5_file.create_dataset("train_img",       train_img_shape, img_dtype)
hdf5_file.create_dataset("train_img_mean",  train_img_shape[1:], np.float32)
hdf5_file.create_dataset("train_imu",       train_imu_shape, np.float32)
hdf5_file.create_dataset("labels",          label_shape,np.float32)

i = 0

for record_path in ordered_path_to_jsons:

    with open(record_path, 'r') as fp:
        json_data = json.load(fp)

    basepath = os.path.dirname(record_path)
    image_filename = json_data["cam/image_array"]
    image_path = os.path.join(basepath, image_filename)

    angle    = float(json_data['user/angle'])
    throttle = float(json_data["user/throttle"])

    img_arr = load_scaled_image_arr(image_path, cfg)

    hdf5_file["train_img"][i, ...]  = img_arr[None]
    hdf5_file["labels"][i,...]      = np.array([angle, throttle])

    try:
        # see if an IMU was used during recording of traning data
        accl_x = float(json_data['imu/acl_x'])
        accl_y = float(json_data['imu/acl_y'])
        accl_z = float(json_data['imu/acl_z'])

        gyro_x = float(json_data['imu/gyr_x'])
        gyro_y = float(json_data['imu/gyr_y'])
        gyro_z = float(json_data['imu/gyr_z'])

        hdf5_file["train_imu"][i,...] = np.array([accl_x, accl_y, accl_z, gyro_x, gyro_y, gyro_z],dtype=np.float32)
    except:
        hdf5_file["train_imu"][i,...] = np.zeros(6,dtype=np.float32)

    mean_img += img_arr / float(n_records)

hdf5_file["train_img_mean"][...] = mean_img
hdf5_file.close()
print('Finished processing all files')

mean_uint_img = np.asarray(mean_img,dtype=img_dtype)

plt.imshow(mean_uint_img)
plt.show()





