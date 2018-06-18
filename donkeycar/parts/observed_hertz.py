import time

class ObservedHertz:
    def __init__(self,display=False):
        '''
        Track how many actual cycles are processed by the vehicle loop rather
        than the hoped for rate set with config.DRIVE_LOOP_HZ
        Useful for determining if a model is too complex for Raspberry Pi
        
        Is run on a non-threaded basis so that the counter is only incremented
        when called in the run-loop of the vehicle object.
        
        For the first second it will return hz = 0
        '''
        self.start_time = time.time()
        self.cycle_count = 0
        self.hz = 0
        # should this part display the current hz on  stdout
        self.print_hz = display
        self.on = True

    def update(self):
        '''
        This is a blocking call in order to only update once per 
        vehicle object loop. 
        :return: 
        '''
        current_time = time.time()
        if current_time - self.start_time <= 1.0:
            # less than 1 second has passed, increment the counter
            self.cycle_count += 1
        else:
            # 1 second has passed, set the hz to the current counter
            # and reset count and time.
            self.hz = self.cycle_count
            self.cycle_count = 1
            self.start_time = current_time

    def run(self):
        self.update()
        if self.print_hz:
            print('Current hz: ~',self.hz)
        return self.hz

    def shutdown(self):
        self.on = False
        print('stopping hertz monitor')
