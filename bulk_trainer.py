import donkeycar.train as dkt
import donkeycar.config as dkconfig
import os
import numpy as np
import pickle
from time import time
'''
This is a bulk trainer that allows me to run multiple training sessions
on my specific machine. You will need to adapt it extensively for your own
use. I'm running two GTX-1070 GPUs on a 6-core i7 Ubuntu 16.04

definitions of each experiment are held in an importable 'e_def' function
which provides a dictionary of setting for the experiements. Because I have two 
GPU, I split my experiments up based on batch size. I typically train at both 64
and 128. By altering the gpu_idx between 0 & 1 for each run, I can put the 64 batch
size experiment on one GPU and the 128 batch size experiment on another. 

This also requires my modifications to the Keras.py part and relies upon using
my DataGenerator class.

By altering what I import as 'exp_n', I can also run separate instances for seperate
experiements or split my list of experiments over separate processes for a poor-mans 
multi-processing. 

Using my sepecfic gear, I can usually run 4 experiments at a time to reduce overall
training time.

If you are going to fix your experiments at a certain batch size, then it seems reasonable
to rework this to split the work based on some other criteria. Perhaps you want to run a 
'augmented' vs 'non_augmented' at the same time. Or perhaps you just want to run 4
totally different experiements. ?? Up to you to figure out how to rework this.. :)
'''

# from testing_files.exp_13_aug import e_def as exp_n
from testing_files.training_experiments.exp13b import e_def as exp_n

# from testing_files.exp8 import e_def as exp_n
# from testing_files.exp9 import e_def as exp_n

# on my set up I have GPUs at ID 0 and 2. This provides an index
# into 'gpu_ids' list below.
gpu_idx             = 0
experiment_date     = '2018-07-27'  # folder into which experiment will be stored
hist_pkl_name       = 'exp_13b'    # name for resulting history file of experiment
num_runs            = 5             # number of training sessions to run per experiment.

# tubs of data to use
imu_tubs = ['../d2IMU/data/smoothed_imu/2018-07-08_3n_smooth_run_1',
            '../d2IMU/data/smoothed_imu/2018-07-09_imu_run_2']

tub_names   = ','.join(imu_tubs)

# if augmenting data, which augmentations to perform. Color Balance and Noise are
# most expensive. clean_percent is the percentage of training data that is not
# augmented per batch on average. If batch_size = 64, clean_percent 0.10 = ~6.4 samples
# are not augmented in the batch.
aug_args = {'vary_color_balance':True,'vary_sharpness':False,'vary_bright':True,
            'vary_contrast':True, 'add_noise':True,'vary_sat':True,
            'clean_percent':0.15}

path_for_training_config = '../d2IMU/config.py' # which config file to use as base
cfg = dkconfig.load_config(path_for_training_config)
cfg.MAX_EPOCHS = 30

# these are the NVIDIA gpu ids that show up when I run
# echo $CUDA_VISIBLE_DEVICES.
gpu_ids     = ['gpu-0','gpu-2']

try:
    os.stat('models')
except:
    os.mkdir('models')

experiments = exp_n(tub_names)

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

        experiment['batch_size'] = bs   # lets move away from using hardwired batch_size
        cfg.BATCH_SIZE           = bs   # original project stores this a config option.

        try:
            os.stat(base_name+'/'+str(bs))
        except:
            os.mkdir(base_name+'/'+str(bs))

        for i in range(1,(num_runs+1)):
            # set the GPU ID so that we can run separate experiements on different GPUs and balance
            # the training load.
            experiment['gpu'] = gpu_ids[gpu_idx]

            for aug in [0,1]:
                if (aug == 1):
                    # indicate on the model name that data augmentation was used.
                    aug_text = '_aug'
                    # aug_args could be moved into the exp files itself to facilitate looping
                    # over multiple various augmentations.
                    experiment['aug_args'] = aug_args
                    experiment['no_aug_percent'] = aug_args['clean_percent']
                    experiment['aug'] = True
                else:
                    aug_text = ''
                    experiment['aug'] = False

                model_file_name = base_name +'/'+ str(bs)+'/'+model_base_name + '_'+str(experiment['gpu'])+aug_text+'_v'+str(i)
                experiment['model_name'] = model_file_name

                start_time = time()
                history = dkt.multi_train(cfg, **experiment)
                end_time = time()

                print ('Time for Experiment',experiment['exp'],':',end_time-start_time)
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