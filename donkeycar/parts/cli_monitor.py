import time

class MonitorCLI:
    def __init__(self, delay=1, display=True, *args,**kwargs):
        '''
        Display data collected on the command line. Items like IMU values,
        current throttle, angle, actual hz, etc.
        :param delay: how often to show the hard return data in second.
        :param display: wether to display the data or no. Default is true since that's the point of this part
        :param args: void
        :param kwargs: dict of formats and value name. {'hard_return:[[[{'format':'throttle: {0:1.3f}','data':'user/throttle'}],{next item on row.}]
                                                                       [second hard row]]]
                                                        'last_line':[[[{format:'accel x: {0:1.3f},'data':'imu/accelx'}, {etc}]]]}
        '''
        self.start_time = time.time()
        # how often to display debug info. Generally speaking once per second. Set up this
        # part with cfg.DRIVE_LOOP_HZ to match them up.
        self.delay = delay
        self.cycle_count = 0
        self.display_items = kwargs
        # should this part display the current hz on  stdout
        self.show_debug = display
        self.on = True

    def run(self,*data):
        if self.show_debug:
            hard_return = self.display_items.get('hard_return') # each row of array is printable row
            last_line   = self.display_items.get('last_line')   # should only be one row of items

            # data in the 'hard_return' list is displayed every n seconds
            #TODO: need to advance array pointer to offset the occasional from constant data.
            current_time = time.time()
            if current_time - self.start_time >= self.delay:
                self.start_time = current_time
                if hard_return in locals():
                    print("\n" + "-" * 50)
                    for row in hard_return:
                        row_str = ''
                        for item in row:
                            row_str += item['format'].format(data[item['data']])
                            row_str += ' '
                        print(row_str)

            # data in the last line is displayed based on runtime Hz speed.
            try:
                zip_list = zip(last_line,data)
                row_str = ' '.join([a[0].format(a[1]) for a in zip_list])
                print(row_str,end='\r')
            except:
                pass

    def shutdown(self):
        self.on = False
        print('Stoping CLI monitor')
