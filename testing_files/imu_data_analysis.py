import matplotlib.pyplot as plt

import numpy as np
from pprint import pprint
np.set_printoptions(precision=4,suppress=True,linewidth=150)

import pickle

pickle_open = False
try:
    file = open('imu_data.pkl', 'rb')
    imu_data_set = pickle.load(file)
except:
    file = open('imu_data.pkl', 'wb')
    pickle_open = True

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

    pickle.dump(imu_data_set,file)
    file.close()

num_records = len(imu_data_set)
print('Len set', num_records)

mean_imu = np.mean(imu_data_set,axis=0)
std_imu  = np.std(imu_data_set,axis=0)
min_imu  = np.min(imu_data_set,axis=0)
max_imu = np.max(imu_data_set,axis=0)
key_names = ['steering','throttle', 'accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z']
print(key_names)
print('Mean Values:\t',mean_imu)
print('Standard Dev:\t',std_imu)
print('Min Values:  \t',min_imu)
print('Max Values:  \t',max_imu)

bin_spread = 0.20
# build a figure that has the 5 primary gyroscope values. We ignore the accel-z axis
plt.figure(dpi=300)

ax_x  = plt.subplot2grid((2,3),(0,0))
# For now, set to 1 since we're using normed values and have 'thresholded' data.
# after collecting clean data, 'top' can be set to something more reasonable like 0.40 or the like
ax_x.set_ylim(bottom=0,top=1)
# the minimum x and y accel seem to be around [-5,5]
ax_x.set_xlim(left=-5,right=5)

ax_y  = plt.subplot2grid((2,3),(0,2))
ax_y.set_ylim(bottom=0,top=1)
ax_y.set_xlim(left=-5,right=5)

g_x  = plt.subplot2grid((2,3),(1,0))
g_x.set_ylim(bottom=0,top=0.2)
# gyro data is in degrees/sec rather than g-force ??
# we might need to mean normalize all the gyro data for better training???
g_x.set_xlim(left=-20,right=20)

g_y  = plt.subplot2grid((2,3),(1,1))
g_y.set_ylim(bottom=0,top=0.2)
g_y.set_xlim(left=-20,right=20)

g_z  = plt.subplot2grid((2,3),(1,2))
g_z.set_ylim(bottom=0,top=0.05)
# the z gyro measures the rotation angle when turning, it displays a distinct multi-modal
# distribution
g_z.set_xlim(left=-80,right=80)

# expand the number of bins by a divisor
gyro_bins = np.arange(start=min_imu[2],stop=max_imu[2],step=bin_spread/2.0)
print ('accel-x #bins:',len(gyro_bins))
ax_x.hist(x=imu_data_set[:,2],bins=gyro_bins,density=True)
x0,x1 = ax_x.get_xlim()
# this forces the aspect ratio of the histogram to be square. Needs modifier is y-axis is
# not 1.0
ax_x.set_aspect(abs(x1)+abs(x0))

gyro_bins = np.arange(start=min_imu[3],stop=max_imu[3],step=bin_spread/2.0)
print ('accel-y #bins:',len(gyro_bins))
ax_y.hist(x=imu_data_set[:,3],bins=gyro_bins,density=True)
x0,x1 = ax_y.get_xlim()
ax_y.set_aspect(abs(x1)+abs(x0))

# gyro data plots
# reduce the number of bins by a multiplier.
gyro_bins = np.arange(start=min_imu[5],stop=max_imu[5],step=bin_spread*2)
print ('gyro-x #bins:',len(gyro_bins))

g_x.hist(x=imu_data_set[:,5],bins=gyro_bins,density=True)
x0,x1 = g_x.get_xlim()
# here we have to modify the aspect ration to get a square box.
g_x.set_aspect((abs(x1)+abs(x0))/0.20)

gyro_bins = np.arange(start=min_imu[6],stop=max_imu[6],step=bin_spread*2)
print ('gyro-y #bins:',len(gyro_bins))
g_y.hist(x=imu_data_set[:,6],bins=gyro_bins,density=True)
x0,x1 = g_y.get_xlim()
g_y.set_aspect((abs(x1)+abs(x0))/0.20)

gyro_bins = np.arange(start=min_imu[7],stop=max_imu[7],step=bin_spread*4)
print ('gyro-z #bins:',len(gyro_bins))

g_z.hist(x=imu_data_set[:,7],bins=gyro_bins,density=True)
x0,x1 = g_z.get_xlim()
g_z.set_aspect((abs(x1)+abs(x0))/.05)

ax_x.set_xlabel('g-force accel-x')
ax_y.set_xlabel('g-force accel-y')
g_x.set_xlabel('gyro-x')
g_y.set_xlabel('gyro-y')
g_z.set_xlabel('gyro-z')
plt.show(block=True)
