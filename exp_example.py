'''
Set of experiment as control for same image model but no IMU model
'''


def e_def(tub_name_string):
    # create a easy to access set of keras layers. These just happen to be Dense,
    # but can be any Keras layer
    model_funcs = {'dense14': "Dense(14, activation='relu')",
                   'dense7': "Dense(7, activation='relu')",
                   'dense4': "Dense(4, activation='relu')",
                   'drop10': "Dropout(.1)",
                   'drop25': "Dropout(.25)",
                   'drop50': "Dropout(.50)"}

    # build a set of experiments. In this case I'm building experiments for the IMU model
    # you need to use something similar to my Keras.py file which allows models to be
    # dynamically adjusted for each experiment.
    imu_model_def = ['dense14', 'drop10', 'dense14']
    imu_smaller_model = []
    for f in imu_model_def:
        imu_smaller_model.append(model_funcs[f])

    imu_model_def = ['dense14','drop10','dense7','drop10','dense4']
    imu_14x7x4 =[]
    for f in imu_model_def:
        imu_14x7x4.append(model_funcs[f])

    imu_model_def = ['dense7','drop10','dense7']
    imu_7x7 =[]
    for f in imu_model_def:
        imu_7x7.append(model_funcs[f])

    # now set up a set of experiments. 'exp' become the identifier for the experiements.
    # Thus '19' = Experiment 19 in my book, but you can use anything
    experiments = [
        {'exp':19,
                    # model_base_name is essentially the name of the experiment
                    'model_base_name':'def_linear_no_imu',
                    # model_type is used by utils.get_model_by_type
                    'model_type':'imu_lin',
                    # bs is batch_size, index 0 will train on GPU 1 for sets 128 & 32
                    # GPU 2 would train 64 & 256. This is very specific to the fact
                    # that I have two GPUs on my system.
                    'bs': [[128,32],[64,256]],
                    # model_function is model definition in keras.py part. Allows you to vary
                    # your model definitions
                    'model_function':'default_lin_no_imu',
                    # my donkeycar has an MPU6050 accel/Gryro.
                    'imu_inputs':6,
                    # the pickle cache file will be created the first time a set of tubs is
                    # read and then subsequently used. If you change your tubs, you need to
                    # change this as well. Saves a bunch of time by avoiding the expensive
                    # collate_records call
                    'pkl_cache':'2018-07_08_imu_data.pkl',
                    # sting of tub names joined with commas. see bulk_trainer.py
                    'tubs':tub_name_string,
                    # should augmentation occur? See bulk_trainer for augmentation settings
                    # as well as augment.py, datagenerator.py, and train.py
                    'aug':False},

        {'exp': 21,
                     'model_base_name': 'imu_14_d10_14_5imu',
                     # i'm only training a 128 & 64 batch size, for this experiment.
                     'model_type': 'imu_lin', 'bs': [[128], [64]],
                     'model_function': 'default_imu',
                     'imu_inputs': 5,
                     # this experiment uses only a portion of the IMU data, ignoring the accel_z axis
                     'imu_pick': ['accl_x', 'accl_y', 'gyro_x', 'gyro_y', 'gyro_z'],
                     # uses a different model than the experiment above, this is a custom IMU model
                     # that 'default_imu' in keras will dynamically assemble.
                     'imu_model': imu_smaller_model,
                     # uses a different picke file because we want only 5 of 6 IMU data points.
                     'pkl_cache': '2018-07_08_imu_data_5pt.pkl',
                     'tubs': tub_name_string,
                     # augmentation will occur based on the aug_args in bulk_trainer.py
                     'aug': True},
    ]
    return experiments