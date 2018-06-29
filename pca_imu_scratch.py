import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from pprint import pprint
np.set_printoptions(precision=4,suppress=True,linewidth=150)


from sklearn import decomposition
from sklearn import datasets
#import donkeycar as dk

import donkeycar.train as dkt
import donkeycar.config as dkconfig


path_for_training_config = '../d2IMU/config.py'
tub_names = '../d2IMU/data/double_line_IMU/2018-03-16_late_afternoon_320-240'

cfg = dkconfig.load_config(path_for_training_config)


print('Gathering Records')
records = dkt.gather_records(cfg, tub_names)

gen_records ={}
opts = {}
opts['pickle_file'] = '2018-03-16_late_afternoon_320-240.pkl'
opts['categorical'] = False
opts['val_split']   = 1 - cfg.TRAIN_TEST_SPLIT

dkt.collate_records(records, gen_records, opts)
record_keys = list(gen_records.keys())
record_keys = sorted(record_keys)

num_records = len(record_keys)
i = 0
imu_data_set = np.zeros(shape=(num_records, 8))


for k in record_keys:
    imu_data_set[i,0]  = gen_records[k]['angle']
    imu_data_set[i,1]  = gen_records[k]['throttle']
    imu_data_set[i, 2:] = gen_records[k]['imu_array']
    i += 1

print('Len set', num_records)

mean_imu = np.mean(imu_data_set,axis=0)
std_imu  = np.std(imu_data_set,axis=0)
min_imu  = np.min(imu_data_set,axis=0)
max_imu = np.max(imu_data_set,axis=0)

pprint('accel, throttle, accl_x, accl_y, accl_z, gyro_x, gyro_y, gyro_z')
pprint(mean_imu)
pprint(std_imu)
pprint(min_imu)
pprint(max_imu)

i=0
for d in imu_data_set:
#    if d[0]>-.01 and d[0]<.01 and d[1]>.84:
    if d[6] > 18 or d[6] < -18:
        print('-'*25)
        print(imu_data_set[i-2:i+3,:],record_keys[i],gen_records[record_keys[i]]['image_path'])
    i+=1
