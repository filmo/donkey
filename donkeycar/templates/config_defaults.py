""" 
CAR CONFIG 

This file is read by your car application's manage.py script to change the car
performance. 

EXMAPLE
-----------
import dk
cfg = dk.load_config(config_path='~/d2/config.py')
print(cfg.CAMERA_RESOLUTION)

"""


import os

#PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

#VEHICLE
DRIVE_LOOP_HZ = 20
MAX_LOOPS = 100000

#CAMERA
#CAMERA_RESOLUTION = (128, 160) #(height, width)
CAMERA_RESOLUTION = (240, 320)
CAMERA_TYPE = "PICAM"   # (PICAM|WEBCAM|CVCAM)
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
CAMERA_FRAMERATE = DRIVE_LOOP_HZ

#STEERING
STEERING_CHANNEL = 1
STEERING_LEFT_PWM =  440 #420
STEERING_RIGHT_PWM = 290 #360

#THROTTLE
THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 400
THROTTLE_STOPPED_PWM = 360
THROTTLE_REVERSE_PWM = 310


#TRAINING
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.8
MAX_EPOCHS = 100
SHOW_PLOT = True
VEBOSE_TRAIN = True
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 5
MIN_DELTA = .0005
PRINT_MODEL_SUMMARY = True      #print layers and weights to stdout
OPTIMIZER = None                #adam, sgd, rmsprop, etc.. None accepts default
LEARNING_RATE = 0.001           #only used when OPTIMIZER specified
LEARNING_RATE_DECAY = 0.0       #only used when OPTIMIZER specified

#model transfer options
FREEZE_LAYERS = False
NUM_LAST_LAYERS_TO_TRAIN = 7

#RNN or 3D
SEQUENCE_LENGTH = 3

#JOYSTICK
USE_JOYSTICK_AS_DEFAULT = False
JOYSTICK_MAX_THROTTLE = 0.3
JOYSTICK_STEERING_SCALE = 1.0
AUTO_RECORD_ON_THROTTLE = True
CONTROLLER_TYPE='ps3' #(ps3|ps4)
USE_NETWORKED_JS = False
NETWORK_JS_SERVER_IP = "192.168.0.1"


AUTO_RECORD_ON_THROTTLE = True

#RC Controller (2-channel stock controller for now) - alpha!! 2018-01-02
USE_RC_CONTROLLER   = True
RC_MAX_THROTTLE     = 1.00
# RC_STEERING_SCALE: +1 or -1 depending on transmitter. On mine I needed -1 to map 'right turn'
# on controller to 'right turn' on DonkeyCar (so that it matches the webcontroller interface
RC_STEERING_SCALE   = -1
# these are determined emperically based on your controller and car. Drive your car and run
# raw_pulse_feed.py. Once you have your steering and throttle trim set on the controller,
# make a note of the numbers listed. It's listed [throttle,steering]. This will be your 'center' value
# RC_HIGH is maximum value at full, forward throttle & full Left/Right  (depends on controller)
# RC_LOW is minimum value at full reverse throttle & full Right/Left (depends on controller)
# RC_DEAD is how much tollerance you want at 'center' in order to consider it 'stopped' or
# steering angle = 0, RC_TOLERANCE allows over HIGH and under LOW values to be clampped to LOW or HIGH
# without throwing an error.
RC_CENTER = [1400,1350]
RC_LOW    = [850, 850]
RC_HIGH   = [1950,1930]
RC_DEAD   = 15
RC_TOLERANCE = 100

#IMU
HAVE_IMU = True

