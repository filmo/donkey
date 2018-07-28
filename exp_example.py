'''
Set of experiment as control for same image model but no IMU model
'''


def e_def(imu_tubs):
    model_funcs = {'dense14': "Dense(14, activation='relu')",
                   'dense7': "Dense(7, activation='relu')",
                   'dense4': "Dense(4, activation='relu')",
                   'drop10': "Dropout(.1)",
                   'drop25': "Dropout(.25)",
                   'drop50': "Dropout(.50)"}

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

    experiments = [
        {'exp':19,
                    'model_base_name':'def_linear_no_imu',
                    'model_type':'imu_lin',
                    'bs': [[128,32],[64,256]],
                    'model_function':'default_lin_no_imu',
                    'imu_inputs':6,
                    'pkl_cache':'2018-07_08_imu_data.pkl',
                    'tubs':','.join(imu_tubs),
                    'aug':False},
        {'exp':20,
                    'model_base_name':'def_linear_no_imu',
                    'model_type':'imu_lin', 'bs': [[64],[128]],
                    'model_function':'default_lin_no_imu',
                    'imu_inputs':6,
                    'pkl_cache':'2018-07_08_imu_data.pkl',
                    'tubs':','.join(imu_tubs),
                    'aug':True},

        {'exp': 21,
                     'model_base_name': 'imu_14_d10_14_5imu',
                     'model_type': 'imu_lin', 'bs': [[128], [64]], 'model_function': 'default_imu',
                     'imu_inputs': 5,
                     'imu_pick': ['accl_x', 'accl_y', 'gyro_x', 'gyro_y', 'gyro_z'],
                     'imu_model': imu_smaller_model,
                     'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': True},

        {'exp': 22,
                     'model_base_name': 'imu_14_d10_14_5imu',
                     'model_type': 'imu_lin', 'bs': [[128], [64]], 'model_function': 'default_imu',
                     'imu_inputs': 5,
                     'imu_pick': ['accl_x', 'accl_y', 'gyro_x', 'gyro_y', 'gyro_z'],
                     'imu_model': imu_smaller_model,
                     'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': False},

        {'exp': 23,
                     'model_base_name': 'imu_14_d10_14',
                     'model_type': 'imu_lin', 'bs': [[64], [128]], 'model_function': 'default_imu',
                     'imu_inputs': 6,
                     'imu_model': imu_smaller_model,
                     'pkl_cache': '2018-07_08_imu_data.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': False},

        {'exp': 24,
                     'model_base_name': 'imu_14_d10_14',
                     'model_type': 'imu_lin', 'bs': [[64], [128]], 'model_function': 'default_imu',
                     'imu_inputs': 6,
                     'imu_model': imu_smaller_model,
                     'pkl_cache': '2018-07_08_imu_data.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': True},

        {'exp': 25,
                     'model_base_name': 'imu_14x7x4',
                     'model_type': 'imu_lin', 'bs': [[128], [64]], 'model_function': 'default_imu',
                     'imu_inputs': 6,
                     'imu_model': imu_14x7x4,
                     'pkl_cache': '2018-07_08_imu_data.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': True},
        {'exp': 26,
                     'model_base_name': 'imu_14x7x4',
                     'model_type': 'imu_lin', 'bs': [[64], [128]], 'model_function': 'default_imu',
                     'imu_inputs': 6,
                     'imu_model': imu_14x7x4,
                     'pkl_cache': '2018-07_08_imu_data.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': False},

        {'exp': 27,
                     'model_base_name': 'imu_7x7_5imu',
                     'model_type': 'imu_lin', 'bs': [[128], [64]], 'model_function': 'default_imu',
                     'imu_inputs': 5,
                     'imu_pick': ['accl_x', 'accl_y', 'gyro_x', 'gyro_y', 'gyro_z'],
                     'imu_model': imu_7x7,
                     'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': False},

        {'exp': 28,
                     'model_base_name': 'imu_7x7_5imu',
                     'model_type': 'imu_lin', 'bs': [[128], [64]], 'model_function': 'default_imu',
                     'imu_inputs': 5,
                     'imu_pick': ['accl_x', 'accl_y', 'gyro_x', 'gyro_y', 'gyro_z'],
                     'imu_model': imu_7x7,
                     'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': True},

    ]
    return experiments