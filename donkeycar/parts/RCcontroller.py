'''
Basic Controller for using RC transmitter that comes with Magnet Car.

Warning: no where near as developed as the PS3/PS4 Joystick controller.

You'll need an Arduino or Teensy. Might as well get a Teensy as it runs at 76 MHz and can handle the interrupts
with a bit more precision than a 16MHz Arduino can. However, it seems to work fine with even old boards like
an Arduino Duemilanove.

Install the provided RC_two_channel.ino onto the board of your choice. The default setting use Digital Pins 2 & 3
The RC Receiver on the Magnet car will need power. This can be provided by running power from the Arduino or Teensy
into either the 'Bat' port or Channel 3 or 4 which are currently unused.

Run wires from the Recievers data pin for Channels 1 & 2 in to Digital Pins 2 & 3 on the Arduino

Attach the Arduino or Teensy to the Raspberry Pi via a USB cable.

Run the following to activate using the RC controller.

donkey createcar --path ~/d2RC --template donkeyRC.py

python manage.py drive --rc

You can open the web page controller in order to see the video image. This is totally hacky right now.

TODO: rework the web controller to gracefully interface with the RC controller part as a selectable Control Mode.
TODO: right now, I'm simply hijacking it in order to see the video while using the RC transmitter.

TODO: Lots of open questions. (1) Integrate into the webcontroller so that additional button actions can be handled by
TODO: the webpage while the RC controller handles throttle and angle. I have zero experience with python tornado web app
TODO: library that it seems to currently use. :(
TODO: (2) extend library to run with more sophisticated RC controller? Perhaps just wait to see what jwatte comes up with??
TODO: (3) turning on/off the recording seems dependent only on throttle and not on vehicle dynamics. Seems highly problematic
TODO: for getting accurate training data. IMU or other type of odometry seem like good extensions to help mitigate this
TODO: issue.

See also work by jwatte at https://github.com/jwatte/donkey_racing. He is working on a much more sophisticated Teensy RC
controller. I suspect it will ultimately be much better.
'''

import time
import serial
from threading import Thread

# import for syntactical ease
from donkeycar.parts.web_controller.web import LocalWebController
class MultiChannelRC():
    '''
    Define a 5 channel receiver.

    This is specific to Phil Glau's build. Throttle, Steering, milliAmps, milliVolts, watts

    This was written for Teensy 3.2 but could work on Arduino Uno. You will need to correctly ID the device as different
    Arduino show up in /dev/ with different names.

    Arduino Uno, Due, Teensy:   /dev/ttyACM0
    Arduino Duemilanove:        /dev/ttyUSB0
    '''

    def __init__(self,
                    dev_fn,
                    cfg,
                    # deadband needs to be at least high enough to prevent 'auto-record' from
                    # triggering on spurious signals. In reality my car doesn't even begin to move
                    # until it's gone over 0.70 and the slowest it will move is around 0.57
                    deadband=15,
                    tolerance=100):
        self.dev_fn = dev_fn    # right now '/dev/ttyUSB0' corresponds with to Arduino Deminulova
        self.serial_fd = None
        self.low = cfg.RC_LOW   # set in the config file. Each car will be a bit different.
        self.high = cfg.RC_HIGH
        self.center = cfg.RC_CENTER
        self.deadband = deadband
        self.tolerance = tolerance
        self.num_channels = 5
        # used in mapIn to only map the throttle and steering to -1+1 floats.
        self.num_pulse_channels = 2
        # ordering is important on outputs and output_types!!
        self.outputs        = ['user/throttle','user/angle', 'car/amps','car/volts','car/watts','void/mode', 'recording']
        self.output_types   = ['float','float','int','int','float','str','bool']

    def init(self):
        print('5 Channel RC: Opening Microcontroller via Serial: %s...' % self.dev_fn, )
        try:
            self.serial_fd = serial.Serial(self.dev_fn, 57600)
            self.serial_fd.isOpen()
            print('5 Channel RC: Microcontroller opened!')
        except IOError:
            self.serial_fd.close()
            self.serial_fd.open()
            print('5 Channel RC: port was already open. Closed and reopened')
        return [0 for x in range(self.num_channels)]

    def getLatestStatus(self):
        ''' This clears the buffer and returns only the last value

            This is faster than polling the arduino for the latest value. Thus the arduino just stuffs values
            into the serial buffer as fast as it's set to and then this will disregard all except the most
            recent value.

            ### pretty 'hacky' ###
        '''
        status = b''  # an empty byte string
        if (not self.serial_fd.isOpen()):
            print("5 Channel RC: **** Serial Port is not open anymore !! ****")
            time.sleep(5)

        while self.serial_fd.inWaiting() > 0:
            # read and discard any values except the most recent.
            # when Arduino Hz = 2x python Hz this results in 1 to 2
            # discarded results. Because the Arduino is sending data with a '\n' at the
            # end of each transmision, each 'line' in the serial buffer corresponds to a
            # single data point. The most recent one is the last one in the buffer.
            status = self.serial_fd.readline()

        # raw format = b'1234 1234\r\n' from the Arduino
        # and needs to be decoded and split into an array.
        try:
            vel_angle_raw = status.strip().decode("utf-8").split(" ")
        except Exception:
            # unable to decode the values from the Arduino. Typically happens
            # at startup when the serial connection is being started and lasts
            # a few cycles. Junk in the trunk..
            vel_angle_raw = ['0' for x in range(self.num_channels)]

            # map the string values to integers.
        if len(vel_angle_raw) == self.num_channels:
            # this used to be int, but needs to be float because watts is a float from the Teensy.
            # could convert watts to milliWatts and return from Teensy as int if it turns out that
            # int is needed.
            return list(map(float, vel_angle_raw))
        else:
            return [0 for x in range(self.num_channels)]

    def mapIn(self, usValue_ary):
        '''
        Takes a tuple of int values representing usPulses per channel and
        converts it to a float range between -1 and +1 per channel

        :param usValue_ary: an array of INTs [1360,1350] for example representing [throttle,steering]
        :return: an array of floats like [-0.23, 0.75]
        '''
        return_value = [0 for x in range(self.num_pulse_channels)]
        for i in range(self.num_pulse_channels):
            value = usValue_ary[i]
            if (value < self.low[i]):
                # the value is less than the low value.
                if (value < (self.low[i] - self.tolerance)):
                    # when initially starting up, the arduino will send '0's, so
                    # the first N sample are below the low tolerance.

                    # the value is below even the tolerance value
                    # return 0 as NOP
                    return_value[i] = 0
                    continue
                # else return the thresholded lower value
                value = self.low[i]

            if (value > self.high[i]):
                # likewise, if the value is higher than the defined high, check to see
                # if it is within a tolerance
                if (value > (self.high[i] + self.tolerance)):
                    return_value[i] = 0
                    continue
                # otherwise return the thresholded high value
                value = self.high[i]

            # now our pulses are constrained to a range defined by the per channel HIGH and LOW usPulse values
            # Lets convert them to a -1 to + 1 range compatible with DonkeyCar
            lowDeadBand = self.center[i] - self.deadband
            highDeadBand = self.center[i] + self.deadband

            if (value < lowDeadBand):
                # convert to a value between -1 and 0
                return_value[i] = (value - lowDeadBand) / float(lowDeadBand - self.low[i])
            elif (value > highDeadBand):
                # convert to a vlaue between 0 and +1
                return_value[i] = (value - highDeadBand) / float(self.high[i] - highDeadBand)
            else:
                # else we're in the deadband, just return 0
                return_value[i] = 0
        # return the tuple of converted values.
        return return_value

    def poll(self):
        '''
        Read the serial buffer and converts the values to an array of floats
        :return:
        '''
        raw_data_ary = self.getLatestStatus()

        # set some fail-safe values in case nothing is returned.
        float_commands = [0.0 for x in range(self.num_channels)]
        if len(raw_data_ary) == self.num_channels:
            # if there are n values, convert the INT usPulses to Float values
            # only convert the first two ints to [-1 +1] for steering and throttle.
            # index 3,4,5 are milliAmps, milliVolts and Watts respectively
            float_commands = self.mapIn(raw_data_ary[:self.num_pulse_channels])

        # add the amps,volts,watts to the return values.
        float_commands += raw_data_ary[self.num_pulse_channels:]
        return float_commands

class TwoChannelRC():
    '''
    Define a basic 2-channel receiver. This is the typical default 'kit' controller that comes with the DonkeyCar
    chassis. Only Steering and Throttle. This is not super robust. You'll need to pre-calibrate your car with the
    calibrate_arduino_usPulse.py script to learn your center values and optionally your 'high' and 'low'. Getting the
    right 'center' values is more crucial as this defines the 'stopped' state of throttle and 'straight ahead' for the
    steering. Your center values will likely be different than mine

    This was written for Arduino but also works with a Teensy. You will need to correctly ID the device as different
    Arduino show up in /dev/ with different names.

    Arduino Uno, Due, Teensy:   /dev/ttyACM0
    Arduino Duemilanove:        /dev/ttyUSB0

    In theory, the controller can be expanded to a N-channel controller such that you could pick up multiple channels
    off of the receiver including buttons and switches. I think it might be best to just copy this Receiver Class
    and name it appropriately for your particular receiver/transmistter and then just adjust RC_Controller as need be.
    See the work done on Joystick controller for switch/button handling as a jumping off point. - Phil Glau 2018-01-02
    '''

    def __init__(self,
                    dev_fn,
                    cfg,
                    # deadband needs to be at least high enough to prevent 'auto-record' from
                    # triggering on spurious signals. In reality my car doesn't even begin to move
                    # until it's gone over 0.70 and the slowest it will move is around 0.57
                    deadband=15,
                    tolerance=100):
        self.dev_fn = dev_fn    # right now '/dev/ttyUSB0' corresponds with to Arduino Deminulova
        self.serial_fd = None
        self.low = cfg.RC_LOW   # set in the config file. Each car will be a bit different.
        self.high = cfg.RC_HIGH
        self.center = cfg.RC_CENTER
        self.deadband = deadband
        self.tolerance = tolerance
        self.num_channels = 2
        self.num_pulse_channels = 2
        # ordering is important on outputs and output_types!!
        self.outputs        = [ 'user/throttle', 'user/angle','void/mode', 'recording']
        self.output_types   = ['float','float','str','bool']

    def init(self):
        print('TwoChannelRC: Opening Microcontroller via Serial: %s...' % self.dev_fn, )
        try:
            self.serial_fd = serial.Serial(self.dev_fn, 57600)
            self.serial_fd.isOpen()
            print('TwoChannelRC: Microcontroller opened!')
        except IOError:
            self.serial_fd.close()
            self.serial_fd.open()
            print('TwoChannelRC: port was already open. Closed and reopened')
        return [0 for x in range(self.num_channels)]

    def getLatestStatus(self):
        ''' This clears the buffer and returns only the last value

            This is faster than polling the arduino for the latest value. Thus the arduino just stuffs values
            into the serial buffer as fast as it's set to and then this will disregard all except the most
            recent value.

            ### pretty 'hacky' ###
        '''
        status = b''  # an empty byte string
        if (not self.serial_fd.isOpen()):
            print("TwoChannelRC: **** Serial Port is not open anymore !! ****")
            time.sleep(5)

        while self.serial_fd.inWaiting() > 0:
            # read and discard any values except the most recent.
            # when Arduino Hz = 2x python Hz this results in 1 to 2
            # discarded results. Because the Arduino is sending data with a '\n' at the
            # end of each transmision, each 'line' in the serial buffer corresponds to a
            # single data point. The most recent one is the last one in the buffer.
            status = self.serial_fd.readline()

        # raw format = b'1234 1234\r\n' from the Arduino
        # and needs to be decoded and split into an array.
        try:
            vel_angle_raw = status.strip().decode("utf-8").split(" ")
        except Exception:
            # unable to decode the values from the Arduino. Typically happens
            # at startup when the serial connection is being started and lasts
            # a few cycles. Junk in the trunk..
            vel_angle_raw = ['0' for x in range(self.num_channels)]

            # map the string values to integers.
        if len(vel_angle_raw) == self.num_channels:
            return list(map(int, vel_angle_raw))
        else:
            return [0 for x in range(self.num_channels)]

    def mapIn(self, usValue_ary):
        '''
        Takes a tuple of int values representing usPulses per channel and
        converts it to a float range between -1 and +1 per channel

        :param usValue_ary: an array of INTs [1360,1350] for example representing [throttle,steering]
        :return: an array of floats like [-0.23, 0.75]
        '''
        return_value = [0 for x in range(self.num_channels)]
        for i in range(self.num_channels):
            value = usValue_ary[i]
            if (value < self.low[i]):
                # the value is less than the low value.
                if (value < (self.low[i] - self.tolerance)):
                    # when initially starting up, the arduino will send '0's, so
                    # the first N sample are below the low tolerance.

                    # the value is below even the tolerance value
                    # return 0 as NOP
                    return_value[i] = 0
                    continue
                # else return the thresholded lower value
                value = self.low[i]

            if (value > self.high[i]):
                # likewise, if the value is higher than the defined high, check to see
                # if it is within a tolerance
                if (value > (self.high[i] + self.tolerance)):
                    return_value[i] = 0
                    continue
                # otherwise return the thresholded high value
                value = self.high[i]

            # now our pulses are constrained to a range defined by the per channel HIGH and LOW usPulse values
            # Lets convert them to a -1 to + 1 range compatible with DonkeyCar
            lowDeadBand = self.center[i] - self.deadband
            highDeadBand = self.center[i] + self.deadband

            if (value < lowDeadBand):
                # convert to a value between -1 and 0
                return_value[i] = (value - lowDeadBand) / float(lowDeadBand - self.low[i])
            elif (value > highDeadBand):
                # convert to a vlaue between 0 and +1
                return_value[i] = (value - highDeadBand) / float(self.high[i] - highDeadBand)
            else:
                # else we're in the deadband, just return 0
                return_value[i] = 0
        # return the tuple of converted values.
        return return_value

    def poll(self):
        '''
        Read the serial buffer and converts the values to an array of floats
        :return:
        '''
        usPulse_ary = self.getLatestStatus()

        # set some fail-safe values in case nothing is returned.
        float_commands = [0.0 for x in range(self.num_channels)]
        if len(usPulse_ary) == self.num_channels:
            # if there are n values, convert the INT usPulses to Float values
            float_commands = self.mapIn(usPulse_ary)

        return float_commands

class RC_Controller(object):
    '''
    Use 2.4ghz RC reciever/transmitter that comes with chassis as an input device. The one I got came
    with a 4ch receiver and 2ch transmitter (throttle & steering)

    Arduino sketch should poll at a higher Hz than this class's polling delay. I'm using 40Hz on the Arduino
    and polling at 25Hz while Donkey Car is running at 20Hz. Seems to work okay for me. If you poll higher or
    the same as the Arduino, you'll end up with lots of [0,0] commands corresponding to missing data.
    There's probably some elegant way to reuse/extrapolate from the prior N recieved commands to fill in small gaps,
    but its easier to just have the Arduino provide slightly more data than the Pi will use. -- Phil 2018-01-02
    '''

    def __init__(self, cfg,
                 poll_delay=0.04, # slightly faster than the frame refresh. Arduino should be at even higher hz. (I use 40hz on arduino)'
                 max_throttle=1.0,
                 steering_scale= -1.0, # on my RC transmitter I needed to flip to -1 this to match manual steering.
                 throttle_scale=1.0,
                 dev_fn='/dev/ttyACM0',
                 auto_record_on_throttle=True,
                 show_cmd=False,
                 cntr = TwoChannelRC
                 ):

        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.poll_delay = poll_delay
        self.running = True
        self.max_throttle = max_throttle
        self.steering_scale = steering_scale
        self.throttle_scale = throttle_scale
        self.recording = False
        self.record_minimum = 0.20
        self.constant_throttle = False
        self.auto_record_on_throttle = auto_record_on_throttle
        self.delay_stop = 5
        self.delay_stop_count = 0
        self.dev_fn = dev_fn
        self.ctrl_class = cntr
        self.rc_controller = None
        self.show_commands = show_cmd
        self.cfg = cfg  # reference to the configuration variables. This is where the RC channels are defined.
        self.return_values = None
        self.outputs = None
        self.output_types = None
        self.num_channels = None
        # perform initiaization right away so that manage.py can access the number of channels
        # and ouput information
        self.init_rc()

    def on_throttle_changes(self):
        '''
        turn on recording when non zero throttle in the user mode.
        '''
        if self.auto_record_on_throttle:
            #TODO: set a tighter tolerance on throttle for recording images??
            #TODO: my RC car stalls out at any throttle less than 0.57 and won't break
            #TODO: standstill until 0.70

            # TODO: should we cut off recording when throttle falls below a certain low threshold?
            # TODO: this may be problematic for dynamics as the car may be moving during deceleration even
            # TODO: if throttle has been released. For example, one can still be steering through a curve
            # TODO: even though the throttle has been released. ??
            # TODO: Limitation of using throttle as a proxy for velocity rather than odometry.
            # self.recording = (abs(self.throttle) < self.record_minimum and self.mode == 'user')

            throttle_state = (abs(self.throttle) > self.record_minimum and self.mode == 'user')

            # wait 5 frames to turn off recording to account for inertia. There are plenty of times
            # where the throttle is '0' but we're actually still driving.
            if throttle_state == False and self.recording and self.delay_stop_count < self.delay_stop:
                self.delay_stop_count += 1
                if self.show_commands:
                    print("--> Stopping soon: ", self.delay_stop_count)
                return self.recording

            # reset delay counter
            if throttle_state:
                self.delay_stop_count = 0

            if self.show_commands and self.delay_stop_count < self.delay_stop:
                print("RC Controller: recording = ",throttle_state)

            self.recording = throttle_state

    def init_rc(self):
        '''
        attempt to init rc controller
        '''
        print('**** inside RCcontroller init_rc *****')
        if self.rc_controller is None:
            try:
                self.rc_controller      = self.ctrl_class(self.dev_fn,cfg=self.cfg)
                self.return_values      = self.rc_controller.init()
                self.outputs            = self.rc_controller.outputs
                self.output_types       = self.rc_controller.output_types
                self.num_channels       = self.rc_controller.num_channels
            except FileNotFoundError:
                print(self.dev_fn, "RC Controller: microcontroller not found.")
                self.rc_controller = None
        return self.rc_controller is not None

    def update(self):
        '''
        poll rc receiver for pulses from the RC transmitter. Default chassis has 2 channels, steering and throttle
        '''

        # wait for Arduino to be online
        while self.running and not self.init_rc():
            time.sleep(5)

        while self.running:
            normalized_values_ary = self.rc_controller.poll()

            throttle = normalized_values_ary[self.cfg.THROTTLE_CHANNEL]  # 0
            steering = normalized_values_ary[self.cfg.STEERING_CHANNEL]  # 1

            self.angle = self.steering_scale * steering

            # this value is often reversed, with positive value when pulling down
            # set throttle_scale to + or - 1 as needed.
            self.throttle = (self.throttle_scale * throttle * self.max_throttle)

            # set the first 2 values of self.return_values to throttle and steering
            # and remainder to any left over values
            self.return_values[self.cfg.THROTTLE_CHANNEL] = self.throttle
            self.return_values[self.cfg.STEERING_CHANNEL] = self.angle
            self.return_values[2:] = normalized_values_ary[2:]

            if self.show_commands:
                print('RC controller: throttle', self.throttle, "angle", self.angle)

            self.on_throttle_changes()

            # sleep for some number of ms to allow Arduino to refill buffer. This is needed!!!
            # otherwise the threaded python script will poll far more frequently than
            # the Arduino is sending serial data to be read.
            time.sleep(self.poll_delay)

    def run_threaded(self, img_arr=None):
        self.img_arr = img_arr

        return self.return_values + [self.mode, self.recording]

    def run(self, img_arr=None):
        raise Exception("We expect for this part to be run with the threaded=True argument.")

    def shutdown(self):
        self.running = False
        time.sleep(0.5)