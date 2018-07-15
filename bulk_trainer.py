import donkeycar.train as dkt
import donkeycar.config as dkconfig
import os
import numpy as np
import pickle
from pprint import pprint
from exp1 import e_def as exp_n
from exp2 import e_def as exp_n
from exp3 import e_def as exp_n
from exp4 import e_def as exp_n
from exp5 import e_def as exp_n

path_for_training_config = '../d2IMU/config.py'

cfg = dkconfig.load_config(path_for_training_config)

# tubs = ['../d2IMU/data/double_line/tub_double_line',
#         '../d2IMU/data/double_line/tub_double_line_pt2']
# # this imu data was thresholded
# imu_tubs = ['../d2IMU/data/double_line_IMU/2018-03-16_late_afternoon_320-240',
#             '../d2IMU/data/double_line_IMU/2018-03-10_evening_line_cross']

imu_tubs = ['../d2IMU/data/smoothed_imu/2018-07-08_3n_smooth_run_1',
            '../d2IMU/data/smoothed_imu/2018-07-09_imu_run_2']


gpu_ids     = ['gpu-0','gpu-2']
# all_batches = [[64,128],[32,256]] # one experiment

tub_names   = ','.join(imu_tubs)

try:
    os.stat('models')
except:
    os.mkdir('models')



gpu_idx = 1
experiment_date = '2018-07-15'

experiments = exp_n(imu_tubs=imu_tubs)
hist_pkl_name = 'exp_5'

try:
    os.stat('models/' + experiment_date)
except:
    os.mkdir('models/' + experiment_date)

all_history = {}

for experiment in experiments:

    model_base_name = experiment['model_base_name']
    base_name = 'models/' + experiment_date + '/'+str(experiment['exp'])+'_'+ model_base_name

    try:
        os.stat(base_name)
    except:
        os.mkdir(base_name)

    batch_sizes = experiment['bs'][gpu_idx]
    all_history[experiment['exp']] = {}

    for bs in batch_sizes:

        all_history[experiment['exp']][bs] = {}
        all_history[experiment['exp']][bs]['run'] = {}

        cfg.BATCH_SIZE = bs
        try:
            os.stat(base_name+'/'+str(bs))
        except:
            os.mkdir(base_name+'/'+str(bs))

        for i in range(1,4):
            # set the GPU ID so that we can run separate experiements on different GPUs and balance
            # the training load.
            experiment['gpu'] = gpu_ids[gpu_idx]

            if (experiment['aug']):
                # indicate on the model name that data augmentation was used.
                aug_text = '_aug'
            else:
                aug_text = ''

            model_file_name = base_name +'/'+ str(bs)+'/'+model_base_name + '_'+str(experiment['gpu'])+aug_text+'_v'+str(i)
            experiment['model_name'] = model_file_name

            history = dkt.multi_train(cfg, **experiment)

            hist_order = ['loss', 'val_loss', 'out_0_loss', 'out_1_loss', 'val_out_0_loss', 'val_out_1_loss']

            hist_rows = []
            hist_dict = {}

            # create Row based history. Each row is one Epoch.
            for ho in hist_order:
                hist_rows.append(history.history[ho])
                hist_dict[ho] = history.history[ho]

            np_rows = np.asarray(hist_rows)

            hist = np_rows.T

            min_validation_loss_row = np.argmin(hist[:, 1])
            num_epochs = len(hist)
            best_epoch = hist[min_validation_loss_row, :]
            min_val_loss = best_epoch[1]

            data = {'history': hist_dict, 'table': hist,'exp':experiment['exp'],
                    'bs':bs,'gpu_id':gpu_ids[gpu_idx],'run':i,'aug':experiment['aug'],
                    'epochs':num_epochs,'best_epoch':best_epoch,'val_loss':min_val_loss,
                    'file':model_file_name}

            all_history[experiment['exp']][bs]['run'][i] = data


file = open('models/' + experiment_date +'/'+hist_pkl_name+'_'+gpu_ids[gpu_idx]+'.pkl', 'wb')
pickle.dump(all_history, file)
file.close()