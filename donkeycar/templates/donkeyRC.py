"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    manage.py (drive) [--model=<model>] [--js]  [--rc] [--type=(linear|categorical|rnn|imu_lin|imu_cat|behavior|3d)] [--camera=(single|stereo)]
    manage.py (train) [--tub=<tub1,tub2,..tubn>]  (--model=<model>) [--no_cache] \
		      [--type=(linear|categorical|rnn|imu|behavior|3d)] [--transfer=<tmodel>] [--continuous] [--aug]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --js             Use physical joystick.
    --rc             Use RC controller
    --type MODEL     Set up inputs based on type of model you are running
"""
import os
from docopt import docopt

import donkeycar as dk

# import parts
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.transform import Lambda
# moved to 'utils' for lazy loading.
#from donkeycar.parts.keras import KerasCategorical, KerasIMU
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.imu import Mpu6050
from donkeycar.parts.datastore import TubHandler, TubGroup
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.RCcontroller import RC_Controller
from donkeycar.parts.observed_hertz import ObservedHertz
from donkeycar.parts.cli_monitor import MonitorCLI

from pprint import pprint

def drive(cfg, model_path=None, use_joystick=False, use_rcControl=False,model_type=None):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''

    if model_type is None:
        # this is the default model used for training based only on image and throttle/steering
        model_type = "categorical"

    # Initialize car object
    V = dk.vehicle.Vehicle()

    # start adding parts. Order matters, in output of a part is needed for a subsequent part, it should be
    # ordered first. For example, if you need the camera image for your part, then that part should come
    # after the camera part has been added.
    cam = PiCamera(resolution=cfg.CAMERA_RESOLUTION)
    V.add(cam, outputs=['cam/image_array'], threaded=True)

    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        # modify max_throttle closer to 1.0 to have more power
        # modify steering_scale lower than 1.0 to have less responsive steering
        ctr = JoystickController(throttle_scale=cfg.JOYSTICK_MAX_THROTTLE,
                                 steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                                 auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE,
                                 controller_type=cfg.CONTROLLER_TYPE)

        if cfg.USE_NETWORKED_JS:
            from donkeycar.parts.controller import JoyStickSub
            netwkJs = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
            V.add(netwkJs, threaded=True)
            ctr.js = netwkJs

        V.add(ctr,
              inputs=['cam/image_array'],
              outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
              threaded=True)

    elif use_rcControl or cfg.USE_RC_CONTROLLER:
        # modify max_throttle closer to 1.0 to have more power
        # modify steering_scale lower than 1.0 to have less responsive steering
        # 2018-01-01 very 'alpha' stage.
        rc = RC_Controller(cfg=cfg,
                           max_throttle=cfg.RC_MAX_THROTTLE,
                           steering_scale=cfg.RC_STEERING_SCALE,
                           auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE,
                           show_cmd=False)

        # the RC_Controller takes no inputs from other parts. It uses a serial connection to the
        # microcontroller (Teensy) to receive inputs directly from the MCU.
        V.add(rc,
              # outputs angle and throttle, user/mode is controlled by web interface.
              # the RC part has to be set to 'recording' in order for record_on_throttle to work.
              outputs=['user/angle', 'user/throttle', 'void/mode', 'recording'],
              threaded=True)

        '''
        TODO: I'm pretty sure this is not the correct way to gain access to the web interface
        but its what I managed to do to get it sort of working. Seems like two or more parts
        can't output the same set of outputs. Since I was only using it for the video image it seemed
        to work for my purposes.
        
        What I want to accomplish is have the web interface up to view the video and be able to take
        keyboard input for many of the things that a 2-channel RC controller doesn't allow for while at
        the same time using the RC Transmitter/Receiver to provide the throttle & angle input.
        '''
        ctr = LocalWebController()
        V.add(ctr,
              inputs=['cam/image_array'],
              # the output for angle and throttle go into the void, but
              # recording has to be set to 'recording1' so that it doesn't override
              # the RC controller. angle1, throttle1 and recording1 are unused, but must be specificed
              # as the LocalWebController returns 4 values. could probably also use 'null/angle', etc
              # for clarity if need be.
              outputs=['user/angle1', 'user/throttle1', 'user/mode', 'recording1'],
              threaded=True)

    else:
        # This web controller will create a web server that is capable
        # of managing steering, throttle, and modes, and more.
        ctr = LocalWebController()

        V.add(ctr,
              inputs=['cam/image_array'],
              outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
              threaded=True)

        # See if we should even run the pilot module.

    # This is only needed because the part run_condition only accepts boolean
    def pilot_condition(mode):
        if mode == 'user':
            return False
        else:
            return True

    pilot_condition_part = Lambda(pilot_condition)
    V.add(pilot_condition_part, inputs=['user/mode'], outputs=['run_pilot'])

    #IMU -- even if we're using a model that doesn't incorporate the IMU, we'll still collect
    # the data.
    if cfg.HAVE_IMU:
        print('IMU present')
        # 6-axis IMU
        imu = Mpu6050(show_debug=True)
        # run self calibration to zero out the gyro and accelerometer. The Gyro needs it more than the accel does.
        imu.calibrate()
        # smooth the IMU data by 3 samples
        imu.setSmoothed(use=True,length=3)

        print ('Thresholding IMU:',imu.threshold_zero)
        print ('Smoothing IMU:',imu.use_smoothed)

        V.add(imu, outputs=['imu/acl_x', 'imu/acl_y',
                            'imu/acl_z','imu/gyr_x',
                            'imu/gyr_y', 'imu/gyr_z',
                            'imu/temp'], threaded=True)

    # now we're going to get ready to setup of the DNN based on wether we have an IMU or not.
    if model_type == 'imu_lin' or model_type == 'imu_cat':
        assert (cfg.HAVE_IMU)
        # Run the pilot if the mode is not user.
        inputs = ['cam/image_array',
                  'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                  'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']
    else:
        inputs = ['cam/image_array']

    # Run the pilot if the mode is not user.
    if model_path:
        # TF and Keras will lazy load if a model is passed in. Otherwise, there's no need to fire
        # up TF just to collect training data.
        kwarg = {'model_type':model_type}
        # the model is initialized to whatever is the current definition in Keras,
        kl = dk.utils.get_model_by_type( cfg, **kwarg)
        # however, this load method will overwrite the .model by loading the one found at this
        # path irrespective of whatever was defined above.
        kl.load(model_path)
        # display the model being used.
        kl.model.summary()
        # run_condition acts as a flag the causes the part to only run if the condition is true.
        # in this case 'run_pilot' must be true. (Set by the Lamda 'pilot_condition' part above.)
        V.add(kl,   inputs=inputs,
                    outputs=['pilot/angle', 'pilot/throttle'],
                    run_condition='run_pilot')

    # Choose what inputs should change the car.
    # when user_mode switches between 'user' (human for both), 'local_angle' (angle from Keras model, throttle from human)
    #  or 'local_pilot' (throttle & angle both from Keras model)
    def drive_mode(mode,
                   user_angle, user_throttle,
                   pilot_angle, pilot_throttle):
        if mode == 'user':
            return user_angle, user_throttle

        elif mode == 'local_angle':
            return pilot_angle, user_throttle

        else:
            return pilot_angle, pilot_throttle

    drive_mode_part = Lambda(drive_mode)
    V.add(drive_mode_part,
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'throttle'])

    steering_controller = PCA9685(cfg.STEERING_CHANNEL)
    steering = PWMSteering(controller=steering_controller,
                           left_pulse=cfg.STEERING_LEFT_PWM,
                           right_pulse=cfg.STEERING_RIGHT_PWM)

    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL)
    throttle = PWMThrottle(controller=throttle_controller,
                           max_pulse=cfg.THROTTLE_FORWARD_PWM,
                           zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                           min_pulse=cfg.THROTTLE_REVERSE_PWM)

    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])

    # add tub to save data
    inputs = ['cam/image_array', 'user/angle', 'user/throttle', 'user/mode']
    types  = ['image_array', 'float', 'float', 'str']

    if cfg.HAVE_IMU:
        # if we're collecting data from the IMU, then save it into the tub as well
        inputs += [ 'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                    'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z','imu/temp']

        types +=  ['float', 'float', 'float', 'float', 'float', 'float','float']


        last_line = ["acl_x: {0: 6.3f}","acl_y: {0: 6.3f}","gyr_x: {0: 6.2f}",
                     "acl_y: {0: 6.2f}","acl_z: {0: 7.2f}","temp: {0: 4.1f}",
                     "angle: {0: 5.3f}","throttle: {0: 5.3f}"]

        set = {'last_line':last_line}

        cli_input = ['imu/acl_x', 'imu/acl_y', 'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z','imu/temp','angle','throttle']
        cli_debug = MonitorCLI(**set)
        V.add(part=cli_debug,inputs=cli_input,threaded=False)


    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types)
    V.add(tub, inputs=inputs, run_condition='recording')

    # monitor the actual Hz achieved by the vehicle object. Needs
    # to be run non-threaded to work.
    oz = ObservedHertz(display=False)
    V.add(oz,outputs=['hz'],threaded=False)

    # run the vehicle for 20 seconds
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)

    print("You can now go to <your pi ip address>:8887 to drive your car.")

if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    model_type = args['--type']

    if args['drive']:
        drive(cfg=cfg,
              model_path=args['--model'],
              use_joystick=args['--js'],
              use_rcControl=args['--rc'],
              model_type=model_type)

    if args['train']:
        from donkeycar.train import multi_train
        tub = args['--tub']
        model = args['--model']
        transfer = args['--transfer']
        model_type = args['--type']
        continuous = args['--continuous']
        aug = args['--aug']

        multi_train(cfg, tub, model, transfer, model_type, continuous, aug)