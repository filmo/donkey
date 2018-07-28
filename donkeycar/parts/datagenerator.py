import keras
import numpy as np
import os
import donkeycar.utils as dk_utils
import donkeycar.parts.augment as dk_aug
from random import uniform

class DataGenerator(keras.utils.Sequence):
    '''
     A separate generator needs to be instantiated for Training Data and Validation Data
     The 'records' passed in should contain only Training Data or Validation data but not both. This
     is different than the default train.py generator.

     This does not work with continuous training as the default train.py can.

     npy_cache trades disk space for read speed when opening images. Probably on works well on SSD

     This class largely based on this post:
     https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    '''
    def __init__(self, records, opts, batch_size=64, shuffle=True,train=True, cache=False):
        '''
        Initialize a Datagenerator that conforms to Keras Sequence specifications
        :param records: the original 'gen_records' style dict used in original train.py
        :param opts: dictionary of options. Any cfg. settings should be copied into opts.
        :param batch_size: defaults to 64
        :param shuffle: Shuffle the data.
        :param train: is this generator used for Training or Validation data
        '''
        self.batch_size = batch_size
        # tub_records is the 'gen_records' dictionary. Must contain only Train or Test data
        # not a mix of both!
        self.tub_records = records
        self.opts = opts
        self.shuffle = shuffle
        # initialize the indexes upon initialization.
        self.on_epoch_end()
        self.train_generator = train
        # npy_cache will expand storage requirements. npy frames are stored in the same tubs
        # as the original JPG files. Can feed the GPU faster and use less CPU for augmentation.
        # on my 6-core machine with two GTX-1070s, I can run 4 simultaneous experiments vs
        # 2 to 3 without caching.
        self.npy_cache = True
        self.cache = cache

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.tub_records) // self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # self.indexes contains the keys in to tub_records
        # create a batch of of records. index = 0 to #batchs in epoch
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

        These are set in self.opts

        :param index_key_list: list of keys to the tub_records for a given batch size
        :return: X,y
        '''

        has_imu             = self.opts.get('has_imu')
        imu_inputs          = self.opts.get('imu_inputs')
        has_bvh             = self.opts.get('has_bvh')
        model_out_shape     = self.opts['model_out_shape']

        inputs_img = []
        inputs_imu = []
        inputs_bvh = []
        angles = []
        throttles = []

        if self.opts.get('no_aug_percent'):
            conditional_aug_flag = True
            clean_percent = self.opts.get('no_aug_percent')
        else:
            conditional_aug_flag = False
            clean_percent = 0.0

        for k in index_key_list:
            record = self.tub_records[k]

            basename    = os.path.basename(record['image_path'])
            no_ext      = os.path.splitext(basename)[0] + '.npy'
            npy_path    = (os.path.dirname(record['image_path']) + '/' + no_ext)

            if self.npy_cache:
                # Loading a npy file is ~4x faster than opening via PIL, at the cost of storage space
                # on my 320x240 original image = 7.5KB, npy 160x120 scaled ary = 58KB
                try:
                    # once a correctly sized image exist read it in its native state
                    img_arr = np.load(npy_path)
                except:
                    # first time an image is read, load via PIL and scale as need be.
                    # PIL operations are generally expensive, so do it once and save the results
                    img_arr = dk_utils.load_scaled_image_arr_opt(record['image_path'], self.opts)
                    np.save(npy_path, img_arr, allow_pickle=False)
            else:
                # opens image and returns a numpy array (H,W,n)
                img_arr = dk_utils.load_scaled_image_arr_opt(record['image_path'], self.opts)

            if self.opts['aug'] == True and self.train_generator:
                # use the stored original image and augment it. Only done on training data
                # opts['aug_args'] need to be a dict of augmentation options.
                if conditional_aug_flag:
                    # Train on a mix of clean and augmented data.
                    if (uniform(0, 1) > clean_percent):
                        record['img_data'] = dk_aug.augment_image(img_arr, **self.opts['aug_args'])
                    else:
                        record['img_data'] = img_arr
                else:
                    # augment all data
                    record['img_data'] = dk_aug.augment_image(img_arr, **self.opts['aug_args'])
            else:
                # no augmentation, just return the original image. No validation data is
                # augmented either. Set train = False for a validation generator.
                record['img_data'] = img_arr

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
                                               self.opts['IMAGE_H'],
                                               self.opts['IMAGE_W'],
                                               self.opts['IMAGE_DEPTH'])

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