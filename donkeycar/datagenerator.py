import keras
import numpy as np
import donkeycar.utils as dk_utils
import donkeycar.parts.augment as dk_aug

class DataGenerator(keras.utils.Sequence):
    '''
     A separate generator needs to be instantiated for Training Data and Validation Data
     The 'records' passed in should contain only Traning Data or Validation data but not both. This
     is different than the default train.py generator.
    '''
    def __init__(self, cfg, opts, records, batch_size=32, shuffle=True, *args, **kwargs):
        'Initialization'
        self.batch_size = batch_size
        # tub_records is the 'gen_records' dictionary. Must contain only Train or Test data
        # not a mix of both!
        self.tub_records = records
        self.opts = opts
        self.cfg  = cfg
        self.shuffle = shuffle

        # initialize the indexes upon initialization.
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.tub_records) // self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # self.indexes contains the keys in to tub_records
        # create a batch of of records. index = [0,#batchs in epoch]
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.indexes = np.arange(len(self.tub_records))

        # if not self.indexes:
        #     # only needs to be set up once per DataGenerator ??

        # for donkeycar, the indexes are alphanumerical
        self.indexes = list(self.tub_records.keys())

        if self.shuffle == True:
            # numpy.random.shuffle is in-place
            np.random.shuffle(self.indexes)

    def __data_generation(self, index_key_list):
        '''
        Generates data containing batch_size samples.

        X and y are 2D arrays of Inputs and Outputs
        X can be [[img]] or '[[img],[img]]', or [[img],[bhv]], etc.
        Y is [[float,float]] for linear models or
             [[steering_bin],[throttle_bin]] for categorical models.

        These are set in self.opts and self.cfg

        :param index_key_list: list of keys to the tub_records
        :return: X,y
        '''

        has_imu             = self.opts['has_imu']
        has_bvh             = self.opts['has_bvh']
        model_out_shape     = self.opts['model_out_shape']

        inputs_img = []
        inputs_imu = []
        inputs_bvh = []
        angles = []
        throttles = []

        for k in index_key_list:
            record = self.tub_records[k]

            if record['original_img_data'] is None:
                # image has never been loaded for this record. Load it into memory now.
                img_arr = dk_utils.load_scaled_image_arr(record['image_path'], self.cfg)
                # store as the original image. If later augmented, we can just reference
                # this original rather than re-opening it again.
                record['original_img_data'] = img_arr

            # perform the augmentation on the stored image to avoid the file IO at the cost of
            # memory to store images.
            if self.opts['aug']:
                # use the stored original image and augment it. Only done on training data
                record['img_data'] = dk_aug.augment_image(record['original_img_data'], do_cb=True, do_noise=False)
            else:
                # no augmentation, just return the original image. No validation data is
                # augmented either.
                record['img_data'] = record['original_img_data']

            if has_imu:
                inputs_imu.append(record['imu_array'])

            if has_bvh:
                inputs_bvh.append(record['behavior_arr'])

            inputs_img.append(record['img_data'])
            angles.append(record['angle'])
            throttles.append(record['throttle'])

            # clear out the img_data as it will either be recreated or restored from
            # original_img_data. Saves about 40% memory which can be an issue when
            # doing augmentation on multiple process threads.
            record['img_data'] = None

        # keras data needs to be numpy array
        img_arr = np.array(inputs_img).reshape(self.batch_size,
                                               self.cfg.IMAGE_H,
                                               self.cfg.IMAGE_W,
                                               self.cfg.IMAGE_DEPTH)

        # this should be reworked to be more flexible
        if has_imu:
            X = [img_arr, np.array(inputs_imu)]
        elif has_bvh:
            X = [img_arr, np.array(inputs_bvh)]
        else:
            X = [img_arr]

        if model_out_shape[1] == 2:
            y = [np.array([angles, throttles])]
        else:
            y = [np.array(angles), np.array(throttles)]

        return X,y