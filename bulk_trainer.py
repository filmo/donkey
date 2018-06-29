import donkeycar.train as dkt
import donkeycar.config as dkconfig
import os
from pprint import pprint

path_for_training_config = '../d2IMU/config.py'

cfg = dkconfig.load_config(path_for_training_config)

# tubs = ['../d2IMU/data/double_line/tub_double_line',
#         '../d2IMU/data/double_line/tub_double_line_pt2']

imu_tubs = ['../d2IMU/data/double_line_IMU/2018-03-16_late_afternoon_320-240',
            '../d2IMU/data/double_line_IMU/2018-03-10_evening_line_cross']

gpu_ids     = ['g-0','g-2']
all_batches = [[64,128],[32,256]] # one experiment
all_batches = [[32,256],[64,128]] # second experiment (so as to avoid having two 256-trainings on same GPU at same time

all_batches = [[64],[128]]  # 32 and 256 never seem to be much better than 64 or 128 (2018-06-23)
gpu_idx     = 1

tub_names   = ','.join(imu_tubs)
# tub_names = imu_tubs[0]
# tub_names   = ','.join(tubs)
# tub_names   = tubs[0]

model_base_name = 'lin_no_imu_aug' #(uses IMU linear but cancels out effect of IMU
model_type      = 'imu' # 'categorical' or 'imu' for imu linear
base_name       = 'models/'+model_base_name+'/both_sets'

try:
    os.stat('models')
except:
    os.mkdir('models')

try:
    os.stat('models/'+model_base_name)
except:
    os.mkdir('models/'+model_base_name)

try:
    os.stat(base_name)
except:
    os.mkdir(base_name)

aug         = True
batch_sizes = all_batches[gpu_idx]

for bs in batch_sizes:
    cfg.BATCH_SIZE = bs
    try:
        os.stat(base_name+'/'+str(bs))
    except:
        os.mkdir(base_name+'/'+str(bs))

    for i in range(1,4):
        gpu_num = gpu_ids[gpu_idx]

        if (aug):
            aug_text = '_aug'
        else:
            aug_text = ''

        model_name = base_name +'/'+ str(bs)+'/'+model_base_name + '_gpu-'+str(gpu_num)+aug_text+'_v'+str(i)+'.h5'

        dkt.multi_train(cfg, tub=tub_names, model=model_name,  model_type=model_type, gpu=gpu_num,aug=aug)
