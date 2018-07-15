'''
Set of experiment as control for same image model but no IMU model
'''


def e_def(imu_tubs):
    model_funcs = {'dense14': "Dense(14, activation='relu')",
                   'dense12': "Dense(12, activation='relu')",
                   'dense6': "Dense(6, activation='relu')",
                   'dense10': "Dense(10, activation='relu')",
                   'dense7': "Dense(7, activation='relu')",
                   'dense4': "Dense(4, activation='relu')",
                   'dense3': "Dense(3, activation='relu')",
                   'drop10': "Dropout(.1)",
                   'drop25': "Dropout(.25)",
                   'drop50': "Dropout(.50)"}

    imu_model_def = ['dense14','dense14']
    imu_14x14 = []
    for f in imu_model_def:
        imu_14x14.append(model_funcs[f])

    imu_model_def = ['dense12','drop10','dense6','drop10','dense3']
    imu_12x6x3 =[]
    for f in imu_model_def:
        imu_12x6x3.append(model_funcs[f])

    imu_model_def = ['dense7','drop25','dense4']
    imu_7xd25x4 =[]
    for f in imu_model_def:
        imu_7xd25x4.append(model_funcs[f])

    imu_model_def = ['dense10','drop10','dense7','drop10','dense4']
    imu_10x7x4 =[]
    for f in imu_model_def:
        imu_10x7x4.append(model_funcs[f])

    i = 28

    experiments = [

        {'exp': i +1,
                     'model_base_name': 'imu_14x14_no_DO',
                     'model_type': 'imu_lin', 'bs': [[128,32], [64,256]], 'model_function': 'default_imu',
                     'imu_inputs': 6,
                     'imu_model': imu_14x14,
                     'pkl_cache': '2018-07_08_imu_data.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': True},

        {'exp': i + 2,
                     'model_base_name': 'imu_14x14_no_DO',
                     'model_type': 'imu_lin', 'bs': [[64,256], [128,32]], 'model_function': 'default_imu',
                     'imu_pick': ['accl_x', 'accl_y', 'gyro_x', 'gyro_y', 'gyro_z'],
                     'imu_model': imu_14x14,
                     'pkl_cache': '2018-07_08_imu_data.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': False},

        {'exp': i + 3,
                     'model_base_name': 'imu_12x6x3',
                     'model_type': 'imu_lin', 'bs': [[64,256], [128,32]], 'model_function': 'default_imu',
                     'imu_inputs': 6,
                     'imu_model': imu_12x6x3,
                     'pkl_cache': '2018-07_08_imu_data.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': False},

        {'exp': i + 4,
                     'model_base_name': 'imu_12x6x3',
                     'model_type': 'imu_lin', 'bs': [[128,32], [64,256]], 'model_function': 'default_imu',
                     'imu_inputs': 6,
                     'imu_model': imu_12x6x3,
                     'pkl_cache': '2018-07_08_imu_data.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': True},

        {'exp': i + 5,
                     'model_base_name': 'imu_7xd25x4',
                     'model_type': 'imu_lin', 'bs': [[128,32], [64,256]], 'model_function': 'default_imu',
                     'imu_inputs': 6,
                     'imu_model': imu_7xd25x4,
                     'pkl_cache': '2018-07_08_imu_data.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': True},
        {'exp': i + 6,
                     'model_base_name': 'imu_7xd25x4',
                     'model_type': 'imu_lin', 'bs': [[64,256], [128,32]], 'model_function': 'default_imu',
                     'imu_inputs': 6,
                     'imu_model': imu_7xd25x4,
                     'pkl_cache': '2018-07_08_imu_data.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': False},

        {'exp': i + 7,
                     'model_base_name': 'imu_7xd25x4_5imu',
                     'model_type': 'imu_lin', 'bs': [[64,256], [128,32]], 'model_function': 'default_imu',
                     'imu_inputs': 5,
                     'imu_pick': ['accl_x', 'accl_y', 'gyro_x', 'gyro_y', 'gyro_z'],
                     'imu_model': imu_7xd25x4,
                     'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': False},

        {'exp': i + 8,
                     'model_base_name': 'imu_7xd25x4_5imu',
                     'model_type': 'imu_lin', 'bs': [[128,32], [64,256]], 'model_function': 'default_imu',
                     'imu_inputs': 5,
                     'imu_pick': ['accl_x', 'accl_y', 'gyro_x', 'gyro_y', 'gyro_z'],
                     'imu_model': imu_7xd25x4,
                     'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': True},
        {'exp': i + 9,
                     'model_base_name': 'imu_10x7x4_5imu',
                     'model_type': 'imu_lin', 'bs': [[128,32], [64,256]], 'model_function': 'default_imu',
                     'imu_inputs': 5,
                     'imu_pick': ['accl_x', 'accl_y', 'gyro_x', 'gyro_y', 'gyro_z'],
                     'imu_model': imu_10x7x4,
                     'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                     'aug': False},

        {'exp': i + 10,
                     'model_base_name': 'imu_10x7x4_5imu',
                     'model_type': 'imu_lin', 'bs': [[64,256], [128,32]], 'model_function': 'default_imu',
                     'imu_inputs': 5,
                     'imu_pick': ['accl_x', 'accl_y', 'gyro_x', 'gyro_y', 'gyro_z'],
                     'imu_model': imu_10x7x4,
                     'pkl_cache': '2018-07_08_imu_data_5pt.pkl', 'tubs': ','.join(imu_tubs),
                      'aug': True},
    ]
    return experiments