'''
Set of experiment using 5 values from IMU and a variety of different network
definitions. The IMAGE network remains fixed at default settings.

This eliminate the accel_z which is basically downward gravity.
'''

def e_def(imu_tubs):
    model_funcs = {'dense14':"Dense(14, activation='relu')",
                   'dense7':"Dense(7, activation='relu')",
                   'dense4':"Dense(4, activation='relu')",
                   'drop10':"Dropout(.1)",
                   'drop25':"Dropout(.25)",
                   'drop50':"Dropout(.50)"}

    imu_model_def = ['dense14','drop10','dense14','drop10','dense14']
    imu_def_mod = []
    for f in imu_model_def:
        imu_def_mod.append(model_funcs[f])

    imu_model_def = ['dense14','drop10','dense14']
    imu_smaller_model =[]
    for f in imu_model_def:
        imu_smaller_model.append(model_funcs[f])

    imu_model_def = ['dense7','drop10','dense7','drop10','dense7']
    imu_7x7x7 =[]
    for f in imu_model_def:
        imu_7x7x7.append(model_funcs[f])

    imu_model_def = ['dense14','drop10','dense7','drop10','dense4']
    imu_14x7x4 =[]
    for f in imu_model_def:
        imu_14x7x4.append(model_funcs[f])

    i = 9
    experiments = [

                    {'exp':i+1,
                    'model_base_name':'imu_def_5imu',
                    'model_type':'imu_lin', 'bs': [[64],[128]], 'model_function':'default_imu',
                    'imu_inputs': 5,
                     'imu_pick':['accl_x','accl_y','gyro_x','gyro_y','gyro_z'],
                    'imu_model': imu_def_mod,
                    'pkl_cache':'2018-07_08_imu_data_5pt.pkl', 'tubs':','.join(imu_tubs),
                    'aug':True},

                   {'exp': i+2,
                    'model_base_name': 'imu_14_d10_14_5imu',
                    'model_type': 'imu_lin', 'bs': [[128], [64]], 'model_function': 'default_imu',
                    'imu_inputs': 5,
                    'imu_pick':['accl_x','accl_y','gyro_x','gyro_y','gyro_z'],
                    'imu_model': imu_smaller_model,
                    'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                    'aug': True},

                   {'exp': i+3,
                    'model_base_name': 'imu_14_d10_14_5imu',
                    'model_type': 'imu_lin', 'bs': [[64], [128]], 'model_function': 'default_imu',
                    'imu_inputs': 5,
                    'imu_pick':['accl_x','accl_y','gyro_x','gyro_y','gyro_z'],
                    'imu_model': imu_smaller_model,
                    'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                    'aug': False},

                   {'exp':  i+4,
                    'model_base_name': 'imu_7x7x7_5imu',
                    'model_type': 'imu_lin', 'bs': [[128], [64]], 'model_function': 'default_imu',
                    'imu_inputs': 5,
                    'imu_pick':['accl_x','accl_y','gyro_x','gyro_y','gyro_z'],
                    'imu_model': imu_7x7x7,
                    'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                    'aug': False},

                   {'exp': i+5,
                    'model_base_name': 'imu_7x7x7_5imu',
                    'model_type': 'imu_lin','bs': [[64], [128]], 'model_function': 'default_imu',
                    'imu_inputs': 5,
                    'imu_pick':['accl_x','accl_y','gyro_x','gyro_y','gyro_z'],
                    'imu_model': imu_7x7x7,
                    'pkl_cache': '2018-07_08_imu_data_5pt.pkl',
                    'tubs': ','.join(imu_tubs),
                    'aug': True},

                   {'exp': i+6,
                    'model_base_name': 'imu_14x7x4_5imu',
                    'model_type': 'imu_lin', 'bs': [[128], [64]], 'model_function': 'default_imu',
                    'imu_inputs': 5,
                    'imu_pick':['accl_x','accl_y','gyro_x','gyro_y','gyro_z'],
                    'imu_model': imu_14x7x4,
                    'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                    'aug': True},

                   {'exp': i+7,
                    'model_base_name': 'imu_14x7x4_5imu',
                    'model_type': 'imu_lin', 'bs': [[64], [128]], 'model_function': 'default_imu',
                    'imu_inputs': 5,
                    'imu_pick':['accl_x','accl_y','gyro_x','gyro_y','gyro_z'],
                    'imu_model': imu_14x7x4,
                    'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                    'aug': False},
# moved down here to balance load on GPU/CPUs
                    {'exp': i,
                     'model_base_name': 'imu_def_5imu',
                     'model_type': 'imu_lin', 'bs': [[128], [64]], 'model_function': 'default_imu',
                     'imu_inputs': 5,
                     'imu_pick':['accl_x','accl_y','gyro_x','gyro_y','gyro_z'],
                     'imu_model': imu_def_mod,
                     'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': False},
                   ]
    return experiments