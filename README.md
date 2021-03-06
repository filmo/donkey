# -- Phil Glau fork of project
This contains various bits and pieces I've added to the base project for my Donkeycar
* RC controller: Allows you to use the RC controller that comes with the default RC car base. Needs an Arduino Teensy.
* Various changes to the Keras Models
* Expansion of the data augmentation initially started by Tawn Kramer.
* Hertz monitoring part. Displays true running hz

# donkeycar: a python self driving library 

[![CircleCI](https://circleci.com/gh/wroscoe/donkey.svg?style=svg)](https://circleci.com/gh/wroscoe/donkey)

Donkeycar is minimalist and modular self driving library for Python. It is 
developed for hobbiests and students with a focus on allowing fast experimentation and easy 
community contributions.  

#### Quick Links
* [Donkeycar Updates & Examples](http://donkeycar.com)
* [Build instructions and Software documentation](http://docs.donkeycar.com)
* [Slack / Chat](https://donkey-slackin.herokuapp.com/)

![donkeycar](./docs/assets/build_hardware/donkey2.PNG)

#### Use Donkey if you want to:
* Make an RC car drive its self.
* Compete in self driving races like [DIY Robocars](http://diyrobocars.com)
* Experiment with autopilots, mapping computer vision and neural networks.
* Log sensor data. (images, user inputs, sensor readings) 
* Drive your car via a web or game controler.
* Leverage community contributed driving data.
* Use existing hardware CAD designs for upgrades.

### Getting driving. 
After building a Donkey2 you can turn on your car and go to http://localhost:8887 to drive.

### Modify your cars behavior. 
The donkey car is controlled by running a sequence of events

```python
#Define a vehicle to take and record pictures 10 times per second.

from donkeycar import Vehicle
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.datastore import Tub


V = Vehicle()

#add a camera part
cam = PiCamera()
V.add(cam, outputs=['image'], threaded=True)

#add tub part to record images
tub = Tub(path='~/d2/gettings_started', 
          inputs=['image'], 
          types=['image_array'])
V.add(tub, inputs=['image'])

#start the drive loop at 10 Hz
V.start(rate_hz=10)
```

See [home page](http://donkeycar.com), [docs](http://docs.donkeycar.com) 
or join the [Slack channel](http://www.donkeycar.com/community.html) to learn more.
