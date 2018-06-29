#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Uses the data written by the donkey v2.2 tub writer,
but faster training with proper sampling of distribution over tubs. 
Has settings for continuous training that will look for new files as it trains. 
Modify send_model_to_pi is you wish continuous training to update your pi as it builds.
You can drop this in your ~/d2 dir.
Basic usage should feel familiar: python train.py --model models/mypilot
You might need to do a: pip install scikit-learn


Usage:
    train.py [--tub=<tub1,tub2,..tubn>] (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d)] [--continuous] [--aug]

Options:
    -h --help     Show this screen.    
"""
import os
import glob
import random
import json
from docopt import docopt
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import backend as K
import tensorflow as tf

import donkeycar as dk
from donkeycar.parts.keras import KerasIMU,\
     KerasCategorical, KerasBehavioral, Keras3D_CNN,\
     KerasRNN_LSTM, KerasIMUCategorical

from donkeycar.parts.augment import augment_image
from donkeycar.utils import *

# from sklearn.utils import shuffle
from random import shuffle
from pprint import pprint
np.set_printoptions(precision=4,suppress=True,linewidth=100)



'''
matplotlib can be a pain to setup. So handle the case where it is absent. When present,
use it to generate a plot of training results.
'''
try:
    import matplotlib.pyplot as plt
    do_plot = True
except:
    do_plot = False
    
deterministic = False

if deterministic:
    import random as rn

    print ('****** DETERMINISTIC *******')
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    from keras import backend as K

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

'''
Tub management
'''
def make_key(sample):
    '''
    This is the default key creator. The full path of the tub is used. Index number isn't zero padded.
    :param sample:
    :return:
    '''
    tub_path = sample['tub_path']
    index = sample['index']
    return tub_path + str(index)

def make_ordered_key(sample):
    '''
    make a smaller key. This orders by tub name, then frame index. I'm
    thinking of adding a timestamp to the training data tub format so that we could
    use that instead of tub_name. Index number is zero padded to
    :param sample: dict sample
    :return: reduced size key name 'tub_name-001234'
    '''
    tub_path = sample['tub_path']
    index = sample['index']
    return os.path.basename(tub_path)+'-'+str(index).zfill(8)
#    return str(index).zfill(8)+'-'+os.path.basename(tub_path)

def make_next_key(sample, index_offset):
    '''
    This is used in sequence traning.
    :param sample:
    :param index_offset:
    :return:
    '''
    tub_path = sample['tub_path']
    index = sample['index'] + index_offset
    return tub_path + str(index)

def collate_records(records, gen_records, opts):
    '''
    Passes in gen_records which is modified by reference. (not a deep copy). As a 
    result, it is not retunred
    :param records: list of paths to json records in a tub sorted by frame number
    :param gen_records: declared in calling scope
    :param opts: 
    :return: 
    '''

    if opts['pickle_file']:
        import pickle
        try:
            file = open(opts['pickle_file'],'rb')
            pickled_records = pickle.load(file)
            list_keys = list(pickled_records.keys())
            for k in list_keys:
                gen_records[k] = pickled_records[k]
            print('Returning pickled records')
            return
        except:
            file = open(opts['pickle_file'],'wb')

    count = 0
    for record_path in records:
        basepath = os.path.dirname(record_path)
        index = get_record_index(record_path)
        sample = { 'tub_path' : basepath, "index" : index }

        # key a string of 'path_name'+str(frame number)
        # example: 'path/to/my_data_tub1234' for frame 1234 in tub "my_data_tub"
        # this addition of the tub_path to the beginning of the key allows
        # multiple tubs to have frames with the same numbers without collision.
        #key = make_key(sample)
        key = make_ordered_key(sample)

        if key in gen_records:
            print ('*** WARN *** Found duplicate key:',key)
            continue

        with open(record_path, 'r') as fp:
            # read the data in the json file. record_1234.json for example
            json_data = json.load(fp)

        image_filename  = json_data["cam/image_array"]
        image_path      = os.path.join(basepath, image_filename)

        sample['record_path'] = record_path
        sample["image_path"]  = image_path
        sample["json_data"]   = json_data

        angle    = float(json_data['user/angle'])
        throttle = float(json_data["user/throttle"])

        # print ('before binning: angle',angle,'throttle',throttle)

        if opts['categorical']:
            # this converts the real valued angle and throttle into a binned
            # one-hot vector of approximate ranges.

            # added defaults for angle here for clarity.
            angle    = dk.utils.linear_bin(angle,    N=15, offset=1.0, R=2.0)
            # 2018-06-23. Shift the throttle range to offset= -0.5, R = 0.5.
            # this will clamp any values below 50% throttle to 0 and expand the
            # usable range. Below 50% my DK doesn't move.
            throttle = dk.utils.linear_bin(throttle, N=20, offset=-0.5, R=0.5)
            #throttle = dk.utils.linear_bin(throttle, N=20, offset=0.0, R=1.0)


        sample['angle']     = angle
        sample['throttle']  = throttle
        # print (sample['angle'],"\n",sample['throttle'])
        # count+= 1
        # if count > 50:
        #     exit()
        try:
            # see if an IMU was used during recording of traning data
            accl_x = float(json_data['imu/acl_x'])
            accl_y = float(json_data['imu/acl_y'])
            accl_z = float(json_data['imu/acl_z'])

            gyro_x = float(json_data['imu/gyr_x'])
            gyro_y = float(json_data['imu/gyr_y'])
            gyro_z = float(json_data['imu/gyr_z'])

            sample['imu_array'] = np.array([accl_x, accl_y, accl_z, gyro_x, gyro_y, gyro_z])
        except:
            pass

        try:
            behavior_arr = np.array(json_data['behavior/one_hot_state_array'])
            sample["behavior_arr"] = behavior_arr
        except:
            pass

        sample['img_data'] = None
        sample['original_img_data'] = None

        #now assign test or val
        sample['train'] = (random.uniform(0., 1.0) > opts['val_split'])
        gen_records[key] = sample
    if opts['pickle_file']:
        pickle.dump(gen_records,file)

class MyCPCallback(keras.callbacks.ModelCheckpoint):


    def __init__(self, send_model_cb=None, *args, **kwargs):
        super(MyCPCallback, self).__init__(*args, **kwargs)
        self.reset_best_end_of_epoch = False
        self.send_model_cb = send_model_cb
        self.last_modified_time = None

    def reset_best(self):
        self.reset_best_end_of_epoch = True

    def on_epoch_end(self, epoch, logs=None):
        super(MyCPCallback, self).on_epoch_end(epoch, logs)

        if self.send_model_cb:
            '''
            check whether the file changed and send to the pi
            '''
            filepath = self.filepath.format(epoch=epoch, **logs)
            if os.path.exists(filepath):
                last_modified_time = os.path.getmtime(filepath)
                if self.last_modified_time is None or self.last_modified_time < last_modified_time:
                    self.last_modified_time = last_modified_time
                    self.send_model_cb(filepath)

        '''
        when reset best is set, we want to make sure to run an entire epoch
        before setting our new best on the new total records
        '''        
        if self.reset_best_end_of_epoch:
            self.reset_best_end_of_epoch = False
            self.best = np.Inf

def send_model_to_pi(model_filename):
    #print('sending model to the pi')
    #command = 'scp %s tkramer@pi.local:~/d2/models/contin_train.h5' % model_filename
    #res = os.system(command)
    #print("result:", res)
    pass

def train(cfg, tub_names, model_name, transfer_model, model_type, continuous, aug):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    ''' 

    verbose = cfg.VEBOSE_TRAIN

    if continuous:
        print("continuous training")
    
    gen_records = {}
    opts = {}
    # model is initialized with 'adam' optimizer which can be overridden below.
    kl = get_model_by_type(model_type, cfg=cfg)

    opts['val_split']   = 1 - cfg.TRAIN_TEST_SPLIT
    opts['categorical'] = type(kl) is KerasCategorical or type(kl) is KerasIMUCategorical
    opts['pickle_file'] = 'imu_data_both.pkl'

    print('training with model type', type(kl))

    if transfer_model:
        print('\n--- loading weights from model', transfer_model)
        kl.load(transfer_model)
        print('\n')
        #when transfering models, should we freeze all but the last N layers?
        if cfg.FREEZE_LAYERS:
            num_to_freeze = len(kl.model.layers) - cfg.NUM_LAST_LAYERS_TO_TRAIN 
            print('freezing %d layers' % num_to_freeze)           
            for i in range(num_to_freeze):
                kl.model.layers[i].trainable = False        

    # records is a python list of all the .json records in the tubs
    # sorted by frame number
    print('Gathering Records')
    records = gather_records(cfg, tub_names)

    print('collating %d records ...' % (len(records)))
    # gen_records is defined in this scope and passed by reference and modified
    # directly inside 'collage_records'
    # the keys of gen_record are a concat of tub_path + frame#
    collate_records(records, gen_records, opts)
    record_keys = list(gen_records.keys())

    if cfg.OPTIMIZER:
        print('Setting optimizer to:', cfg.OPTIMIZER)
        kl.set_optimizer(cfg.OPTIMIZER, cfg.LEARNING_RATE, cfg.LEARNING_RATE_DECAY)

    kl.compile()

    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())
    #    exit()
    opts['keras_pilot'] = kl
    opts['continuous']  = continuous
    opts['aug']         = aug

    def generator(save_best, opts, data, batch_size, isTrainSet=True):
        
        num_records = len(data)

        kl = opts['keras_pilot']

        if type(kl.model.output) is list:
            model_out_shape = (2, 1)
        else:
            model_out_shape = kl.model.output.shape

        has_imu = type(kl) is KerasIMU or type(kl) is KerasIMUCategorical
        has_bvh = type(kl) is KerasBehavioral

        while True:

            if isTrainSet and opts['continuous']:
                '''
                When continuous training, we look for new records after each epoch.
                This will add new records to the train and validation set.
                '''
                records = gather_records(cfg, tub_names, opts)
                if len(records) > num_records:
                    collate_records(records, gen_records, opts)
                    new_num_rec = len(data)
                    if new_num_rec > num_records:
                        print('picked up', new_num_rec - num_records, 'new records!')
                        num_records = new_num_rec 
                        save_best.reset_best()

            batch_data = []
            record_used = []


            keys = list(data.keys())

            # random.shuffle is in-place. sklean.shuffle returns the shuffled array !! 2018-06-27
            shuffle(keys)

            kl = opts['keras_pilot']

            if type(kl.model.output) is list:
                model_out_shape = (2, 1)
                #print('model output shape list', kl.model.output)
            else:
                model_out_shape = kl.model.output.shape
                #print ('model output shape',kl.model.output.shape)

            has_imu = type(kl) is KerasIMU or type(kl) is KerasIMUCategorical
            has_bvh = type(kl) is KerasBehavioral


            for key in keys:

                if not key in data:
                    continue

                _record = data[key]

                if _record['train'] != isTrainSet:
                    # this splits the data into 'training' and validation
                    continue

                if key in record_used:
                    # this is to check to see if we're over sampling some records
                    # in theory we should walk through the entire data set and never hit this.
                    # each epoch we'll reshuffle the data and reset the 'used' set.
                    print('** Reusing Index: ', key)

                record_used.append(key)

                # append the training data record to the batch_data list.
                # once we reach a full batch size, we will process the batch below.
                batch_data.append(_record)

                # once a batch of records has been selected, get the images as any pre-processing
                if len(batch_data) == batch_size:
                    inputs_img = []
                    inputs_imu = []
                    inputs_bvh = []
                    angles = []
                    throttles = []

                    for record in batch_data:
                        #get image data if we don't already have it

                        if record['original_img_data'] is None :
                            # image has never been loaded for this record. Load it into memory now.
                            img_arr = load_scaled_image_arr(record['image_path'], cfg)
                            # store as the original image. If later augmented, we can just reference
                            # this original rather than re-opening it again.
                            record['original_img_data'] = img_arr

                        # perform the augmentation on the stored image to avoid the file IO at the cost of
                        # memory to store images.
                        if opts['aug'] and isTrainSet==True:
                            # use the stored original image and augment it. Only done on training data
                            record['img_data'] = augment_image(record['original_img_data'],do_cb=True,do_noise=False)
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

                    img_arr = np.array(inputs_img).reshape(batch_size,
                                                           cfg.IMAGE_H,
                                                           cfg.IMAGE_W,
                                                           cfg.IMAGE_DEPTH)

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

                    yield X, y
                    batch_data = []

    model_path = os.path.expanduser(model_name)

    #checkpoint to save model after each epoch and send best to the pi.
    save_best = MyCPCallback(send_model_cb=send_model_to_pi,
                             filepath=model_path,
                             monitor='val_loss',
                             verbose=verbose,
                             save_best_only=True,
                             mode='min')

    #stop training if the validation error stops improving.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                min_delta=cfg.MIN_DELTA, 
                                                patience=cfg.EARLY_STOP_PATIENCE, 
                                                verbose=verbose, 
                                                mode='auto')

    train_gen = generator(save_best, opts, gen_records, cfg.BATCH_SIZE, True)
    val_gen   = generator(save_best, opts, gen_records, cfg.BATCH_SIZE, False)

    total_records = len(gen_records)

    num_train = 0
    num_val   = 0

    for key, _record in gen_records.items():
        if _record['train'] == True:
            num_train += 1
        else:
            num_val += 1

    print("train: %d, val: %d" % (num_train, num_val))
    print('total records: %d' %(total_records))
    
    if not continuous:
        steps_per_epoch = num_train // cfg.BATCH_SIZE
        val_steps       = num_val // cfg.BATCH_SIZE
    else:
        steps_per_epoch = 100
        val_steps = 10

    print('Batch Size:',cfg.BATCH_SIZE)
    print('steps_per_epoch', steps_per_epoch, 'steps of validation',val_steps)

    if continuous:
        epochs = 100000
    else:
        epochs = cfg.MAX_EPOCHS

    aug = False
    if aug:
        # workers_count = 1
        # use_multiprocessing = False
        workers_count = 1
        use_multiprocessing = True
    else:
        workers_count = 1
        use_multiprocessing = True

    callbacks_list = [save_best]

    if cfg.USE_EARLY_STOP:
        callbacks_list.append(early_stop)

    history = kl.model.fit_generator(
                    train_gen, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=epochs, 
                    verbose=cfg.VEBOSE_TRAIN, 
                    validation_data=val_gen,
                    callbacks=callbacks_list, 
                    validation_steps=val_steps,
                    workers=workers_count,
                    use_multiprocessing=use_multiprocessing)

    print("\n\n----------- Best Eval Loss :%f ---------" % save_best.best)

    if cfg.SHOW_PLOT:
        try:
            if do_plot:
                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss : %f' % save_best.best)
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.savefig(model_path + '_loss_%f.png' % save_best.best)
                plt.show()
            else:
                print("not saving loss graph because matplotlib not set up.")
        except:
            print("problems with loss graph")

def sequence_train(cfg, tub_names, model_name, transfer_model, model_type, continuous, aug=False):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    trains models which take sequence of images
    '''
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    import json

    assert(not continuous)

    print("sequence of images training")

    if model_type == "rnn":
        kl = KerasRNN_LSTM(image_w=cfg.IMAGE_W,
            image_h=cfg.IMAGE_H,
            image_d=cfg.IMAGE_DEPTH,
            seq_length=cfg.SEQUENCE_LENGTH, num_outputs=2)

    elif model_type == "3d":
        kl = Keras3D_CNN(image_w=cfg.IMAGE_W,
            image_h=cfg.IMAGE_H,
            image_d=cfg.IMAGE_DEPTH,
            seq_length=cfg.SEQUENCE_LENGTH,
            num_outputs=2)
    else:
        raise Exception("unknown model type: %s" % model_type)

    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())

    if transfer_model:
        print('\n--- loading weights from model', transfer_model)
        kl.load(transfer_model)
        print('\n')
        #when transfering models, should we freeze all but the last N layers?
        if cfg.FREEZE_LAYERS:
            num_to_freeze = len(kl.model.layers) - cfg.NUM_LAST_LAYERS_TO_TRAIN
            print('freezing %d layers' % num_to_freeze)
            for i in range(num_to_freeze):
                kl.model.layers[i].trainable = False

    tubs = gather_tubs(cfg, tub_names)

    records = []

    for tub in tubs:
        record_paths = glob.glob(os.path.join(tub.path, 'record_*.json'))
        print("Tub:", tub.path, "has", len(record_paths), 'records')

        record_paths.sort(key=get_record_index)
        records += record_paths


    print('collating records')
    gen_records = {}

    for record_path in records:

        with open(record_path, 'r') as fp:
            json_data = json.load(fp)

        basepath = os.path.dirname(record_path)
        image_filename = json_data["cam/image_array"]
        image_path = os.path.join(basepath, image_filename)
        sample = { 'record_path' : record_path, "image_path" : image_path, "json_data" : json_data }

        sample["tub_path"] = basepath
        sample["index"] = get_image_index(image_filename)

        angle = float(json_data['user/angle'])
        throttle = float(json_data["user/throttle"])

        sample['target_output'] = np.array([angle, throttle])

        sample['img_data'] = None

        key = make_key(sample)

        gen_records[key] = sample

    print('collating sequences')

    sequences = []

    for k, sample in gen_records.items():

        seq = []

        for i in range(cfg.SEQUENCE_LENGTH):
            key = make_next_key(sample, i)
            if key in gen_records:
                seq.append(gen_records[key])
            else:
                continue

        if len(seq) != cfg.SEQUENCE_LENGTH:
            continue

        sequences.append(seq)

    #shuffle and split the data
    train_data, val_data  = train_test_split(sequences, shuffle=True, test_size=(1 - cfg.TRAIN_TEST_SPLIT))

    def generator(data, batch_size=cfg.BATCH_SIZE):
        num_records = len(data)

        while True:
            #shuffle again for good measure
            shuffle(data)

            for offset in range(0, num_records, batch_size):
                batch_data = data[offset:offset+batch_size]

                if len(batch_data) != batch_size:
                    break

                b_inputs_img = []
                b_labels = []

                for seq in batch_data:
                    inputs_img = []
                    labels = []
                    for record in seq:
                        #get image data if we don't already have it
                        if record['img_data'] is None:
                            img_arr = load_scaled_image_arr(record['image_path'], cfg)

                            if aug:
                                img_arr = augment_image(img_arr)

                            record['img_data'] = img_arr
                            
                        inputs_img.append(record['img_data'])
                    labels.append(seq[-1]['target_output'])

                    b_inputs_img.append(inputs_img)
                    b_labels.append(labels)

                X = [np.array(b_inputs_img).reshape(batch_size,\
                    cfg.SEQUENCE_LENGTH, cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)]

                y = np.array(b_labels).reshape(batch_size, 2)

                yield X, y

    train_gen = generator(train_data)
    val_gen = generator(val_data)

    model_path = os.path.expanduser(model_name)

    total_records = len(sequences)
    total_train = len(train_data)
    total_val = len(val_data)

    print('train: %d, validation: %d' %(total_train, total_val))
    steps_per_epoch = total_train // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    kl.train(train_gen, 
        val_gen, 
        saved_model_path=model_path,
        steps=steps_per_epoch,
        train_split=cfg.TRAIN_TEST_SPLIT,
        use_early_stop = cfg.USE_EARLY_STOP
    )

def multi_train(cfg, tub, model, transfer=False, model_type='categorical', continuous=False, aug=False, gpu=False):
    '''
    choose the right regime for the given model type
    '''
    if gpu:
        # limit model to single GPU. Allows for training of multiple models each on their own
        # GPU
        void,gpu_id = gpu.split('-')
        print('Visible GPU set to GPU:',gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

    # Prevent TF from allocating all memory on the GPU. This allows multiple models to train on the same
    # GPU at the same time (albeit with a compute penalty.)
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    K.set_session(sess)

    # map the correct traning function onto 'train_fn'.
    train_fn = train
    if model_type == "rnn" or model_type == '3d':
        train_fn = sequence_train

    train_fn(cfg, tub, model, transfer, model_type, continuous, aug)

def augment_data():
    pass

if __name__ == "__main__":
    args = docopt(__doc__)
    cfg = dk.load_config()
    tub = args['--tub']
    model = args['--model']
    transfer = args['--transfer']
    model_type = args['--type']
    continuous = args['--continuous']
    aug = args['--aug']
    gpu = args['--gpu']
    multi_train(cfg, tub, model, transfer, model_type, continuous, aug, gpu)