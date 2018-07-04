import time
import numpy as np

class Mpu6050:
    '''
    Installation:
    sudo apt install python3-smbus
    or
    sudo apt-get install i2c-tools libi2c-dev python-dev python3-dev
    git clone https://github.com/pimoroni/py-smbus.git
    cd py-smbus/library
    python setup.py build
    sudo python setup.py install

    pip install mpu6050-raspberrypi
    '''

    def __init__(self, addr=0x68, poll_delay=0.0166,show_debug=False):
        from mpu6050 import mpu6050
        self.sensor = mpu6050(addr)
        self.accel = { 'x' : 0., 'y' : 0., 'z' : 0. }
        self.gyro = { 'x' : 0., 'y' : 0., 'z' : 0. }
        self.temp = 0.
        self.poll_delay = poll_delay
        self.on = True
        self.debug = show_debug
        self.threshold_zero = False
        self.accel_thres = 0.30
        self.gyro_thres  = 0.10
        self.history_length = 3
        self.hist = np.zeros(shape=(self.history_length,7),dtype=np.float32)
        self.hist_idx = 0
        self.use_smoothed = False

    def calibrate(self):
        '''
        Perfrom zero-mean calibration. This is dependent on the mpu6050 class which has
        been updated by Phil Glau to perform the calibration. Setting the flag caused the
        mpu6050 clas to use this calibration when returning values

        https://github.com/filmo/mpu6050/tree/soft_calibration
        :return:
        '''
        self.sensor.zero_mean_calibration()
        self.sensor.set_calibrated_flag()

    def threshold_clamp(self):
        '''
        Clamp values to supress noise on IMU.
        :return: void
        '''
        if abs(self.accel['x']) <= self.accel_thres:
            self.accel['x'] = 0.0
        if abs(self.accel['y']) <= self.accel_thres:
            self.accel['y'] = 0.0
        if abs(self.gyro['x']) <= self.gyro_thres:
            self.gyro['x'] = 0.0
        if abs(self.gyro['y']) <= self.gyro_thres:
            self.gyro['y'] = 0.0
        if abs(self.gyro['z']) <= self.gyro_thres:
            self.gyro['z'] = 0.0

    def update(self):
        '''
        When run as a threaded part, this will poll the MPU-6050 every
        n seconds. The default rate is 60hz
        :return: Void
        '''
        while self.on:
            self.poll()
            time.sleep(self.poll_delay)
                
    def poll(self):
        self.accel, self.gyro, self.temp = self.sensor.get_all_data()
        data = np.asarray([self.accel['x'], self.accel['y'],self.accel['z'],
                            self.gyro['x'], self.gyro['y'], self.gyro['z'],self.temp])
        self.hist[self.hist_idx % self.history_length,...] = data
        self.hist_idx += 1

    def run_threaded(self):

        if self.threshold_zero:
            self.threshold_clamp()

        if self.debug:
            # set to '\n' if you want each reading on a new line.
            if self.use_smoothed:
                smoothed = np.mean(self.hist, axis=0)
                print("x = %6.3f, y = %6.3f, gx = %6.3f, gy = %6.3f, gz=%6.3f" %
                      (smoothed[0], smoothed[1], smoothed[3], smoothed[4], smoothed[5],), end='\r')
            else:
                print("x = %6.3f, y = %6.3f, gx = %6.3f, gy = %6.3f, gz=%6.3f" %
                  (self.accel['x'], self.accel['y'], self.gyro['x'], self.gyro['y'], self.gyro['z']),end='\r')

        self.return_values()

    def run(self):
        self.poll()
        self.return_values()

    def setSmoothed(self,use=True,length=3):
        '''
        Set flag and reinitialize queue. With TRUE a n-lenght mean is returned where
        length is number of sample to average over.
        :param use: boolean TRUE for mean values FALSE for instantaneous
        :param lenght: number of samples to average over as INT
        :return: void
        '''
        self.use_smoothed = use
        self.history_length = length
        # reset the smoothing queue as size may have changed.
        self.hist = np.zeros(shape=(self.history_length,7),dtype=np.float32)

    def return_values(self):
        if self.use_smoothed:
            smoothed = np.mean(self.hist, axis=0)
            return smoothed.tolist()
        else:
            return self.accel['x'], self.accel['y'], self.accel['z'], \
                   self.gyro['x'], self.gyro['y'], self.gyro['z'], self.temp

    def shutdown(self):
        self.on = False


if __name__ == "__main__":
    iter = 0
    p = Mpu6050()
    while iter < 100:
        data = p.run()
        print(data)
        time.sleep(0.1)
        iter += 1
     