'''

pilots.py

Methods to create, use, save and load pilots. Pilots 
contain the highlevel logic used to determine the angle
and throttle of a vehicle. Pilots can include one or more 
models to help direct the vehicles motion.


2018-02-20 pulled from https://github.com/tawnkramer/donkey/tree/master/donkeycar/parts
This has a bunch of different models.

It also 'bins' the throttle rather than leaving it linear as in the original code.
'''

import numpy as np

import keras
#from tensorflow.python import keras
from keras import backend as K

import donkeycar as dk

class KerasPilot(object):
    def __init__(self):
        self.model = None
        # default to the 'adam' optimizer. This is used when initially compiling
        # a model. It can later be overridden with the set_optimizer command
        self.optimizer = "adam"
 
    def load(self, model_path):
        self.model = keras.models.load_model(model_path)

    def shutdown(self):
        pass

    def compile(self):
        pass

    def set_optimizer(self, optimizer_type, rate, decay):
        '''
        This allows you to override the default which is 'adam'
        :param optimizer_type: adam,sgd,rmsprop
        :param rate: learning rate
        :param decay: learning decay (when applicable)
        :return: null
        '''
        if optimizer_type == "adam":
            self.model.optimizer = keras.optimizers.Adam(lr=rate, decay=decay)
        elif optimizer_type == "sgd":
            self.model.optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            self.model.optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay)
        else:
            raise Exception("unknown optimizer type: %s" % optimizer_type)
    
    def train(self, train_gen, val_gen, 
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        
        """
        train_gen: generator that yields an array of images and secondary data (like IMU) 
        """

        #checkpoint to save model after each epoch
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path, 
                                                    monitor='val_loss', 
                                                    verbose=verbose, 
                                                    save_best_only=True, 
                                                    mode='min')
        
        #stop training if the validation error stops improving.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   min_delta=min_delta, 
                                                   patience=patience, 
                                                   verbose=verbose, 
                                                   mode='auto')
        
        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)
        
        hist = self.model.fit_generator(
                        train_gen, 
                        steps_per_epoch=steps, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_data=val_gen,
                        callbacks=callbacks_list, 
                        validation_steps=steps*(1.0 - train_split))
        return hist

    def clipped_mse_angle(self,y_true, y_pred):
        '''
        For Linear Models, we may want to clip the predicted value to [-1,+1] for angle
        https://stackoverflow.com/questions/43099233/keras-regression-clip-values
        2018-07 - didn't work very well - Phil
        :param y_true: true steering angle from traning data
        :param y_pred: predicted steeting angle from model
        :return: clipped mse loss
        '''
        return K.mean(K.square(K.clip(y_pred, -1.0, 1.0) - K.clip(y_true, -1.0, 1.0)), axis=-1)

    def clipped_mse_throttle(self,y_true, y_pred):
        '''
        For Linear Models, we may want to clip the predicted value to [0,+1] for throttle
        :param y_true: true throttle from training data
        :param y_pred: predicted throtttle from model.
        :return: clipped mse loss
        '''
        return K.mean(K.square(K.clip(y_pred, 0.0, 1.0) - K.clip(y_true, 0.0, 1.0)), axis=-1)


class KerasCategorical(KerasPilot):
    def __init__(self, input_shape=(120, 160, 3), *args, **kwargs):
        super(KerasCategorical, self).__init__()
        #self.model = default_categorical(input_shape)
        # self.model = default_categorical_ang_lin_throttle(input_shape)
        self.model = default_categorical_ang_and_throttle(input_shape)
        # self.model = smaller_categorical_original(input_shape)
        # self.model = larger_categorical_original(input_shape)
        # self.model = default_categorical_bn(input_shape)
        self.compile()

    def run(self, img_arr):
        if img_arr is None:
            print('no image')
            return 0.0, 0.0

        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle = self.model.predict(img_arr)
        #in order to support older models with linear throttle,
        #we will test for shape of throttle to see if it's the newer
        #binned version.
        throttle_len = len(throttle[0])
        if throttle_len > 1:
            # the model returned a 1-hot vector which needs to be decoded into a scalar
            # to span from 0.0 to 1.0, R needs to be 1.0 and offset = 0.0
            # if you want to limit the range to 0.0 to 0.50, the R would be 0.50
            # throttle = dk.utils.linear_unbin(throttle, N=throttle_len, offset=0.0, R=1.0)

            # 2018-06-23 - clamp values below 50% throttle to 0 and car doesn't move below that
            # speed.
            throttle = dk.utils.linear_unbin(throttle, N = 20, offset = 0.5, R = 0.5)
        else:
            # throttle was a scalar value from the model.
            throttle = throttle[0][0]

        # redundantly checking angle_len and setting defaults, angle_len should be 15 unless
        # otherwise changed during training of the network.
        angle_len = len(angle_binned[0])
        angle = dk.utils.linear_unbin(angle_binned,N=angle_len, offset=-1, R=2.0)

        return angle, throttle

''' use the values in default_categorical_original. tawn is using categorical for 
    throttle as well as angle. 

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                  loss={'angle_out': 'categorical_crossentropy', 
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': 1.0})
'''

class KerasLinear(KerasPilot):
    def __init__(self, num_outputs=2, input_shape=(120, 160, 3), *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        self.model = default_n_linear(num_outputs, input_shape)
        self.compile()

    def compile(self):
        self.model.compile( optimizer=self.optimizer, loss='mse')

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]

class KerasIMU(KerasPilot):
    '''
    A Keras part that take an image and IMU vector as input,
    outputs steering and throttle

    Note: When training, you will need to vectorize the input from the IMU.
    Depending on the names you use for imu records, something like this will work:

    X_keys = ['cam/image_array','imu_array']
    y_keys = ['user/angle', 'user/throttle']
    
    def rt(rec):
        rec['imu_array'] = np.array([ rec['imu/acl_x'], rec['imu/acl_y'], rec['imu/acl_z'],
            rec['imu/gyr_x'], rec['imu/gyr_y'], rec['imu/gyr_z'] ])
        return rec

    kl = KerasIMU()

    tubgroup = TubGroup(tub_names)
    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, record_transform=rt,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)

    '''
    def __init__(self, model=None, num_outputs=2, num_imu_inputs=6, input_shape=(120, 160, 3), *args, **kwargs):
        super(KerasIMU, self).__init__()

        if 'loss_func' in kwargs:
            self.loss_func = kwargs['loss_func']
            print ('Keras loss function set to:',self.loss_func)
        else:
            # default for linear is Mean Squared Error.
            self.loss_func = 'mse'

        if 'imu_inputs' in kwargs:
            # there's a non default number of imu_inputs: example, eliminate accel_z
            self.num_imu_inputs = kwargs['imu_inputs']
        else:
            self.num_imu_inputs = num_imu_inputs

        if 'model_function' in kwargs:
            # set a lambda model_func based on a string name contained in 'model_function' kwargs
            model_func = eval(kwargs['model_function'])
            print ('Using model_function:',kwargs['model_function'])
            self.model = model_func(num_outputs = num_outputs,
                                    num_imu_inputs = self.num_imu_inputs,
                                    input_shape=input_shape,
                                    **kwargs)
        else:
            self.model = default_imu(num_outputs = num_outputs,
                                     num_imu_inputs = self.num_imu_inputs,
                                     input_shape=input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func)

    def run(self, img_arr, *args):
        '''
        accel_x, accel_y, accel_z, gyr_x, gyr_y, gyr_z
        :param img_arr:
        :param args:
        :return:
        '''
        #TODO: would be nice to take a vector input array.
        img_arr = img_arr.reshape((1,) + img_arr.shape)

        # 2018-07-12 - imu data is passed in as *args. This will create the correct set
        # of batches for the IMU data
        imu_arr = np.array(args).reshape(1,len(args))

        outputs = self.model.predict([img_arr, imu_arr])
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]

class KerasIMUCategorical(KerasPilot):
    '''
    A Keras part that take an image and IMU vector as input,
    outputs binned steering and throttle

    Note: When training, you will need to vectorize the input from the IMU.
    Depending on the names you use for imu records, something like this will work:

    '''
    def __init__(self, model=None, num_imu_inputs=6, angle_bins=15,throttle_bins=20, input_shape=(120, 160, 3), *args, **kwargs):
        super(KerasIMUCategorical, self).__init__(*args, **kwargs)
        self.num_imu_inputs = num_imu_inputs
        self.aN = angle_bins
        self.tN = throttle_bins
        self.model= default_imu_categorical(num_imu_inputs=num_imu_inputs,
                                            input_shape=input_shape,
                                            aN=angle_bins,
                                            tN=throttle_bins)
        self.compile()

    def compile(self):
        '''
        Using categorical_crossentropy cross entropy as angle and throttle are one-hot
        vectors rather than scalars. 
        '''
        self.model.compile(optimizer=self.optimizer,
                  loss={'angle_out': 'categorical_crossentropy', 'throttle_out': 'categorical_crossentropy'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': 1.0})

    def run(self, img_arr, accel_x, accel_y, accel_z, gyr_x, gyr_y, gyr_z):
        # TODO: would be nice to take a vector input array.
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        imu_arr = np.array([accel_x, accel_y, accel_z, gyr_x, gyr_y, gyr_z]).reshape(1,self.num_imu_inputs)

        angle_binned, throttle_binned = self.model.predict([img_arr, imu_arr])

        throttle_unbinned = dk.utils.linear_unbin(throttle_binned, N=self.tN, offset=0.0, R=1.0)
        angle_unbinned    = dk.utils.linear_unbin(angle_binned,    N=self.aN, offset=-1,  R=2.0)

        return angle_unbinned,throttle_unbinned

class KerasBehavioral(KerasPilot):
    '''
    A Keras part that take an image and Behavior vector as input,
    outputs steering and throttle
    '''
    def __init__(self, model=None, num_outputs=2, num_behavior_inputs=2 , *args, **kwargs):
        super(KerasBehavioral, self).__init__(*args, **kwargs)
        self.model = default_bhv(num_outputs = num_outputs, num_bvh_inputs = num_behavior_inputs)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')
        
    def run(self, img_arr, state_array):        
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        bhv_arr = np.array(state_array).reshape(1,len(state_array))
        outputs = self.model.predict([img_arr, bhv_arr])
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]

class KerasRNN_LSTM(KerasPilot):
    def __init__(self, image_w=160, image_h=120, image_d=3, seq_length=3, num_outputs=2, *args, **kwargs):
        super(KerasRNN_LSTM, self).__init__(*args, **kwargs)
        image_shape = (image_h, image_w, image_d)
        self.model = rnn_lstm(seq_length=seq_length,
                              num_outputs=num_outputs,
                              image_shape=image_shape)
        self.seq_length = seq_length
        self.image_d = image_d
        self.image_w = image_w
        self.image_h = image_h
        self.img_seq = []
        self.compile()
        self.optimizer = "rmsprop"

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def run(self, img_arr):
        if img_arr.shape[2] == 3 and self.image_d == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq = self.img_seq[1:]
        self.img_seq.append(img_arr)

        img_arr = np.array(self.img_seq).reshape(1, self.seq_length, self.image_h, self.image_w, self.image_d)
        outputs = self.model.predict([img_arr])
        steering = outputs[0][0]
        throttle = outputs[0][1]
        return steering, throttle

class Keras3D_CNN(KerasPilot):
    def __init__(self, image_w=160, image_h=120, image_d=3, seq_length=20, num_outputs=2, *args, **kwargs):
        super(Keras3D_CNN, self).__init__(*args, **kwargs)
        self.model = build_3d_cnn(w=image_w, h=image_h, d=image_d, s=seq_length, num_outputs=num_outputs)
        self.seq_length = seq_length
        self.image_d = image_d
        self.image_w = image_w
        self.image_h = image_h
        self.img_seq = []
        self.compile()

    def compile(self):
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])

    def run(self, img_arr):

        if img_arr.shape[2] == 3 and self.image_d == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq = self.img_seq[1:]
        self.img_seq.append(img_arr)

        img_arr = np.array(self.img_seq).reshape(1, self.seq_length, self.image_h, self.image_w, self.image_d)
        outputs = self.model.predict([img_arr])
        steering = outputs[0][0]
        throttle = outputs[0][1]
        return steering, throttle


def default_categorical_tawn(input_shape=(120, 160, 3),aN=15,tN=20):
    '''
    tawn version of categorical
    :param input_shape: 
    :param aN: 
    :param tN: 
    :return: 
    '''
    from keras.layers import Input
    from keras.models import Model
    from keras.layers import Convolution2D, Cropping2D
    from keras.layers import Dropout, Flatten, Dense

    opt = keras.optimizers.Adam()
    drop = 0.1

    img_in = Input(shape=input_shape, name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Cropping2D(cropping=((30, 0), (0, 0)))(x)  # trim 20 pixels off top

    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Dropout(drop)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Dropout(drop)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    if input_shape[0] > 32 :
        x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    else:
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    if input_shape[0] > 64 :
        x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    elif input_shape[0] > 32 :
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Dropout(drop)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu
    x = Dropout(drop)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(drop)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                     # Classify the data into 50 features, make all negatives 0
    x = Dropout(drop)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    #categorical output of the angle
    # Connect every input with every output and output 15 hidden units.
    # Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    # categories range from -1.0 to 1.0 centered on 0.0
    angle_out = Dense(aN, activation='softmax', name='angle_out')(x)

    #Binned throttle as well. 20 bins as the default from 0.0 to 1.0
    throttle_out = Dense(tN, activation='softmax', name='throttle_out')(x)
    
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    return model

def default_categorical_bn(input_shape=(120, 160, 3)):
    '''
    Categorial Angle with Linear Throttle using BatchNorm between layers.
    :return:
    '''
    from keras.models import Model
    from keras.layers import Input, Dense, BatchNormalization
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Cropping2D

    img_in = Input(shape=input_shape,name='img_in')  # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Cropping2D(cropping=((30, 0), (0, 0)))(x)  # trim 30 pixels off top
    x = Convolution2D(24, (5, 5), strides=(2, 2))(x)  # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2))(x)  # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, (5, 5), strides=(2, 2))(x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, (3, 3), strides=(2, 2))(x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1))(x)  # 64 features, 3px3p kernal window, 1wx1h stride, relu
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten(name='flattened')(x)        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)     # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    # categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)  # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    # continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)  # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model

def default_categorical_ang_and_throttle(input_shape=(120, 160, 3),aN=15,tN=20):
    '''
    This is the categorical given on the wroscoe master repository
    :return:
    '''
    from keras.models import Model
    from keras.layers import Input, Dense, merge
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Cropping2D

    img_in = Input(shape=input_shape,name='img_in')  # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Cropping2D(cropping=((30, 0), (0, 0)))(x)  # trim 30 pixels off top
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)  # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)  # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)  # 64 features, 3px3p kernal window, 1wx1h stride, relu

    x = Flatten(name='flattened')(x)        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)     # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    # categorical output of the angle
    angle_out = Dense(aN, activation='softmax', name='angle_out')(x)  # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    # continous output of throttle

    #Binned throttle as well. 20 bins as the default from 0.0 to 1.0
    throttle_out = Dense(tN, activation='softmax', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    # loss_weights is the portion to which each contribute to the overall loss
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'categorical_crossentropy'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': 0.5})

    return model

def default_categorical_ang_lin_throttle(input_shape=(120, 160, 3)):
    '''
    This is the categorical given on the wroscoe master repository
    :return: 
    '''
    from keras.models import Model
    from keras.layers import Input, Dense, merge
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Cropping2D

    img_in = Input(shape=input_shape,name='img_in')  # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Cropping2D(cropping=((30, 0), (0, 0)))(x)  # trim 30 pixels off top
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(x)  # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(x)  # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)  # 64 features, 3px3p kernal window, 1wx1h stride, relu

    x = Flatten(name='flattened')(x)        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)     # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    # categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)  # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    # continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)  # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model

def larger_categorical_original(input_shape=(120, 160, 3)):
    '''
    This is the categorical given on the wroscoe master repository
    :return:
    '''
    from keras.models import Model
    from keras.layers import Input, Dense, merge
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Cropping2D

    img_in = Input(shape=input_shape,name='img_in')  # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Cropping2D(cropping=((30, 0), (0, 0)))(x)  # trim 30 pixels off top
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)  # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)  # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
#    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Convolution2D(64, (3, 3), strides=(2, 1), activation='relu')(x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(x)  # 64 features, 3px3p kernal window, 1wx1h stride, relu

    x = Flatten(name='flattened')(x)        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)     # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    # categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)  # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    # continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)  # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model



def default_n_linear(num_outputs, input_shape):
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.layers import Convolution2D
    from keras.layers import Dropout, Flatten, Cropping2D

    drop = 0.1
    
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Cropping2D(cropping=((10,0), (0,0)))(x) #trim 10 pixels off top
    #x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    x = Dropout(drop)(x)
    
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = []
    
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))
        
    model = Model(inputs=[img_in], outputs=outputs)
    
    return model


def default_lin_no_imu(num_outputs, num_imu_inputs, input_shape,*args,**kwargs):
    '''
    Notes: this model depends on concatenate which failed on keras < 2.0.8
    This IMU model outputs to scalar values for angle and throttle and
    is and extension of the default_n_linear model
    '''

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.layers import Convolution2D
    from keras.layers import Dropout, Flatten, Cropping2D
    from keras.layers.merge import concatenate

    img_in = Input(shape=input_shape, name='img_in')
    imu_in = Input(shape=(num_imu_inputs,), name="imu_in")

    x = img_in
    if 'img_model' in kwargs:
        for f in kwargs['img_model']:
            keras_func = eval(f)
            x = keras_func(x)
    else:
        x = Cropping2D(cropping=((45,0), (0,0)))(x) #trim 40 pixels off top
        #x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
        x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
        x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
        x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
        x = Flatten(name='flattened')(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(.1)(x)

    #y = imu_in
  #  y = Dense(14, activation='relu')(y)
  #  y = Dropout(.1)(y)
    #    y = Dense(14, activation='relu')(y)
  #  y = Dense(14, activation='relu')(y)

  #  z = concatenate([x, y])
    z = Dense(75, activation='relu')(x)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    outputs = []

    # unlike the categorical model, this uses a purely linear output.
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='out_' + str(i))(z))

    model = Model(inputs=[img_in, imu_in], outputs=outputs)

    return model

def default_imu(num_outputs, num_imu_inputs, input_shape,*args,**kwargs):
    '''
    Notes: this model depends on concatenate which failed on keras < 2.0.8
    This IMU model outputs to scalar values for angle and throttle and
    is and extension of the default_n_linear model
    '''

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.layers import Convolution2D
    from keras.layers import Dropout, Flatten, Cropping2D
    from keras.layers.merge import concatenate
    
    img_in = Input(shape=input_shape, name='img_in')
    imu_in = Input(shape=(num_imu_inputs,), name="imu_in")
    
    x = img_in
    if 'img_model' in kwargs:
        for f in kwargs['img_model']:
            keras_func = eval(f)
            x = keras_func(x)
    else:
        x = Cropping2D(cropping=((45,0), (0,0)))(x) #trim 45 pixels off top
        #x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
        x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
        x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
        x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
        x = Flatten(name='flattened')(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(.1)(x)
    
    y = imu_in

    if 'imu_model' in kwargs:
        print ('Setting custom IMU model')
        # 'imu_model' is a list of keras functions as strings that are evaluated
        # into function and then built attached to the model.
        for f in kwargs['imu_model']:
            print ('IMU Layer:',f)
            keras_func = eval(f)
            y = keras_func(y)
    else:
        # default IMU model
        y = Dense(14, activation='relu')(y)
        y = Dense(14, activation='relu')(y)
        y = Dense(14, activation='relu')(y)

    if 'concat_model' in kwargs:
        for f in kwargs['concat_model']:
            keras_func = eval(f)
            z = keras_func(z)
    else:
        z = concatenate([x, y])
        z = Dense(75, activation='relu')(z)
        z = Dropout(.1)(z)
        z = Dense(50, activation='relu')(z)
        z = Dropout(.1)(z)

    outputs = [] 

    #unlike the categorical model, this uses a purely linear output.
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='out_' + str(i))(z))
        
    model = Model(inputs=[img_in, imu_in], outputs=outputs)
    
    return model

def default_imu_categorical(input_shape=(120, 160, 3),num_imu_inputs = 6, aN=15,tN=20):
    '''
    This is a categorical version for IMU. It borrows model structure from IMU but
    outputs angle and throttle into one-hot vectors of shape aN and tN respectively
    :param num_imu_inputs: defaults to 6 axis IMU. It may be worth zeroing out the accel_z axis as this
                            is just gravity in the downward vector. Seems like a useless expansion of the
                            dimensionality with little reward.
    :param input_shape: size of the image in H x W x D format
    :param aN: Number of bins to create for the steering angle (-1 to +1)
    :param tN: Number of bins to create for the throttle range of 0 to +1. (ignores 'reverse' values)
    :return: keras model that outputs two one-hot vectors
    '''
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers import Dropout, Flatten, Cropping2D
    from keras.layers.merge import concatenate

    img_in = Input(shape=input_shape,       name='img_in')
    imu_in = Input(shape=(num_imu_inputs,), name="imu_in")

    x = img_in
    x = Cropping2D(cropping=((30, 0), (0, 0)))(x)  # trim 35 pixels off top
    # x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
    # on first layer we're going to stride by 3 in the vertical as there is more redundant
    # information than on the horizontal
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)

    y = imu_in
    y = Dense(14, activation='relu')(y)
    y = Dropout(.1)(y)
    y = Dense(14, activation='relu')(y)
#    y = Dense(14, activation='relu')(y)

    z = concatenate([x, y])
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    # categorical output of the angle
    # Connect every input with every output and output 15 hidden units.
    # Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    angle_out    = Dense(aN, activation='softmax', name='angle_out')(z)
    # Binned throttle as well. 20 bins is default
    throttle_out = Dense(tN, activation='softmax', name='throttle_out')(z)

    model = Model(inputs=[img_in, imu_in], outputs=[angle_out, throttle_out])

    return model

def default_bhv(num_outputs, num_bvh_inputs, input_shape):
    '''
    Notes: this model depends on concatenate which failed on keras < 2.0.8
    '''

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.layers import Convolution2D
    from keras.layers import Dropout, Flatten, Cropping2D
    from keras.layers.merge import concatenate
    
    img_in = Input(shape=input_shape, name='img_in')
    bvh_in = Input(shape=(num_bvh_inputs,), name="behavior_in")
    
    x = img_in
    x = Cropping2D(cropping=((60,0), (0,0)))(x) #trim 60 pixels off top
    #x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    
    y = bvh_in
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    
    z = concatenate([x, y])
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    outputs = [] 
    
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='out_' + str(i))(z))
        
    model = Model(inputs=[img_in, bvh_in], outputs=outputs)
    
    return model

def rnn_lstm(seq_length=3, num_outputs=2, image_shape=(120,160,3)):

    from keras.layers import Input, Dense
    from keras.models import Sequential
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers import Dropout, Flatten, Cropping2D
    from keras.layers import LSTM
    from keras.layers.wrappers import TimeDistributed as TD

    img_seq_shape = (seq_length,) + image_shape   
    img_in = Input(batch_shape = img_seq_shape, name='img_in')
    
    x = Sequential()
    x.add(TD(Cropping2D(cropping=((60,0), (0,0))), input_shape=img_seq_shape )) #trim 60 pixels off top
    x.add(TD(Convolution2D(24, (5,5), strides=(2,2), activation='relu')))
    x.add(TD(Convolution2D(32, (5,5), strides=(2,2), activation='relu')))
    x.add(TD(Convolution2D(32, (3,3), strides=(2,2), activation='relu')))
    x.add(TD(Convolution2D(32, (3,3), strides=(1,1), activation='relu')))
    x.add(TD(MaxPooling2D(pool_size=(2, 2))))
    x.add(TD(Flatten(name='flattened')))
    x.add(TD(Dense(100, activation='relu')))
    x.add(TD(Dropout(.1)))
      
    x.add(LSTM(128, return_sequences=True, name="LSTM_seq"))
    x.add(Dropout(.1))
    x.add(LSTM(128, return_sequences=False, name="LSTM_out"))
    x.add(Dropout(.1))
    x.add(Dense(128, activation='relu'))
    x.add(Dropout(.1))
    x.add(Dense(64, activation='relu'))
    x.add(Dense(10, activation='relu'))
    x.add(Dense(num_outputs, activation='linear', name='model_outputs'))
    
    return x

def build_3d_cnn(w, h, d, s, num_outputs):
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Cropping3D

    #Credit: https://github.com/jessecha/DNRacing/blob/master/3D_CNN_Model/model.py
    '''
        w : width
        h : height
        d : depth
        s : n_stacked
    '''
    input_shape=(s, h, w, d)

    model = Sequential()
    #First layer
    model.add(Cropping3D(cropping=((0,0), (50,10), (0,0)), input_shape=input_shape) ) #trim pixels off top
    
    # Second layer

    model.add(Conv3D(
        filters=16, kernel_size=(3,3,3), strides=(1,3,3),
        data_format='channels_last', padding='same')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1,2,2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Third layer
    model.add(Conv3D(
        filters=32, kernel_size=(3,3,3), strides=(1,1,1),
        data_format='channels_last', padding='same')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1, 2, 2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Fourth layer
    model.add(Conv3D(
        filters=64, kernel_size=(3,3,3), strides=(1,1,1),
        data_format='channels_last', padding='same')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1,2,2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Fifth layer
    model.add(Conv3D(
        filters=128, kernel_size=(3,3,3), strides=(1,1,1),
        data_format='channels_last', padding='same')
    )
    model.add(Activation('relu'))
    model.add(MaxPooling3D(
        pool_size=(1,2,2), strides=(1,2,2), padding='valid', data_format=None)
    )
    # Fully connected layer
    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_outputs))
    #model.add(Activation('tanh'))

    return model