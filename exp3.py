'''
Set of experiment as control for same image model but no IMU model
'''

def e_def(imu_tubs):

    i = 17
    experiments = [
        {'exp':i,
                    'model_base_name':'def_linear_no_imu',
                    'model_type':'imu_lin', 'bs': [[128],[64]],
                    'model_function':'default_lin_no_imu',
                    'imu_inputs':6,
                    'pkl_cache':'2018-07_08_imu_data.pkl',
                    'tubs':','.join(imu_tubs),
                    'aug':False},
        {'exp':i+1,
                    'model_base_name':'def_linear_no_imu',
                    'model_type':'imu_lin', 'bs': [[64],[128]],
                    'model_function':'default_lin_no_imu',
                    'imu_inputs':6,
                    'pkl_cache':'2018-07_08_imu_data.pkl',
                    'tubs':','.join(imu_tubs),
                    'aug':True},

                   ]
    return experiments