# PID Controller Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project implements a PID (Proportional Integral Derivative) controller to use with the Udacity car simulator.  With the PID running, the car will autonomously drive around the track, adjusting the steering based on a given CTE (Cross Track Error) value.

![](pid-run.gif)

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

# Project Goals and [Rubric](https://review.udacity.com/#!/rubrics/824/view)

The goals of this project are the following:

* The PID controller must be implemented as was taught in the lessons.
* An implementation of the twiddle algorithm tunes the hyperparameters.
* The vehicle must successfully drive a lap around the track.

# Implementation of the PID controller

In the Udacity car simulator, the CTE value is read from the data message sent by the simulator, and the PID controller updates the error values and predicts the steering angle based on the total error.  This predicted steering angle is a correction of the updated error to the desired setpoint based on proportional, integral, and derivative terms (hence PID).

![image](https://user-images.githubusercontent.com/34095574/88364473-4cbbb000-cd83-11ea-9438-fb0273bf8aff.png)
##### PID Formula (image from Wikipedia)

After the PID calculates the steering angle, a throttle value is derived and sent back to the simulator.  Once a new message is received, the new CTE value is used to start the process of update and prediction again.   


![image](https://user-images.githubusercontent.com/34095574/88364507-68bf5180-cd83-11ea-8e33-a60f77ac9c39.png)


##### PID Process (image from Wikipedia)

The speed at which data messages are sent to the program is highly influenced by the resolution and graphics quality selected in the opening screen of the simulator.  Other factors include speed of the machine running the simulator, the OS and if other programs are competing for CPU/GPU usage.  This is important because I found that if the rate of messages coming into the program were too low, the car would not update fast enough.  It would start oscillating and, eventually, fly off the track.

# Throttle
The other required output value for each data message is a throttle value from -1.0 to 1.0 where -1.0 is 100% reverse and 1.0 is 100% forward.  I decided to use the derived steering value to compute the throttle because:
1) We want to go fast
2) Going fast around corners or while oscillating toward the setpoint usually results in going off the track
3) Going fast when the data message throughput is low also results in going off the track
  
So my throttle algorithm is:
```
if the steering angle is high
    set throttle to 0.25
else
    set throttle to the inverse of the steering value (max 100) normalized to between 0.45 and 1.0
    if the data message throughput rate is low
        lower the throttle value by 0.2
```
I found that using this throttle logic kept the car on the track when oscillating or when the data message throughput was low and allowed for moderately high speeds. 



# Tuning the Hyperparameters

The final hyperparameters were manually chosen. I would tweak P/I/D depending on how the vehicle behaved in the simulation.

For example, if I observed the vehicle was oscillating too harshly about its intended path, I would either decrease P, increase D, or both. On the other hand, if I observed the vehicle going too straight through tight corners, I would do the opposite (increase P, decrease D, or both).

As for the hyperparameter I, I first noticed that setting I to non-zero values was causing the vehicle to oscillate too much in the beginning, sometimes driving off the road at the very start. Then I set I to 0 and the vehicle was able to make it around the track. In the end, I settled for a relative small value of I.


# Results

A short video with the final parameters:

[Link to project video](https://youtu.be/z8wwFCgMu0s)



