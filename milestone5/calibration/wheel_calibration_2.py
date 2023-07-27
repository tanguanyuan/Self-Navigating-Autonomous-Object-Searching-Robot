# for computing the wheel calibration parameters
import numpy as np
import os
import sys
import math
sys.path.insert(0, "../util")
from pibot import Alphabot



def calibrateWheelRadius():
    # Compute the robot scale parameter using a range of wheel velocities.
    # For each wheel velocity, the robot scale parameter can be computed
    # by comparing the time and distance driven to the input wheel velocities.

    ##########################################
    # Feel free to change the range / step
    ##########################################
    
    wheel_velocities_range = range(20, 30, 2)  # or use np.linspace
    delta_times = []

    for wheel_vel in wheel_velocities_range:
        print("Driving at {} ticks/s.".format(wheel_vel))
        # Repeat the test until the correct time is found.
        while True:
            delta_time = input("Input the time to drive in seconds: ")
            try:
                delta_time = float(delta_time)
            except ValueError:
                print("Time must be a number.")
                continue

            # Drive the robot at the given speed for the given time
            ppi.set_velocity([1, 0], tick=wheel_vel, time=delta_time)
            dist = 0.4
            uInput = input("Did the robot travel {}m?[y/N]".format(dist))
            if uInput == 'y':
                delta_times.append(delta_time)
                print("Recording that the robot drove 1m in {:.2f} seconds at wheel speed {}.\n".format(delta_time,
                                                                                                        wheel_vel))
                print('Scale = {:f}'.format(dist/(delta_time*wheel_vel)))
                break

    # Once finished driving, compute the scale parameter by averaging
    num = len(wheel_velocities_range)
    scale = 0
    for delta_time, wheel_vel in zip(delta_times, wheel_velocities_range):
        scale += (dist/(delta_time*wheel_vel))
    scale = scale/num
        # pass # TODO: replace with your code to compute the scale parameter using wheel_vel and delta_time
    print("The scale parameter is estimated as {:.6f} m/ticks.".format(scale))

    return scale


def calibrateBaseline(scale):
    # Compute the robot baseline parameter using a range of wheel velocities.
    # For each wheel velocity, the robot baseline parameter can be computed by
    # comparing the time elapsed and rotation completed to the input wheel
    # velocities to find out the distance between the wheels (wheels_width).

    ##########################################
    # Feel free to change the range / step
    ##########################################
    # range( initial speed, max speed, step speed)
    wheel_vel = 20  # or use np.linspace
    angles_range = range(45, 225, 45)
    delta_times = []

    for angle in angles_range:
        print("Driving for {} degrees.".format(angle))
        # Repeat the test until the correct time is found.
        while True:
            delta_time = input("Input the time to drive in seconds: ")
            try:
                delta_time = float(delta_time)
            except ValueError:
                print("Time must be a number.")
                continue

            # Spin the robot at the given speed for the given time
            ppi.set_velocity([0, 1], tick=20,turning_tick=wheel_vel, time = delta_time)

            angle_rad = angle/180*math.pi
            uInput = input("Did the robot spin {}deg?[y/N]".format(angle))
            if uInput == 'y':
                delta_times.append(delta_time)
                print("Recording that the robot spun 360deg in {:.2f} seconds at wheel speed {}.\n".format(delta_time,wheel_vel))
                print('Baseline = {:f}'.format(scale*delta_time*wheel_vel/(angle_rad/2)))
                break

    # Once finished driving, compute the baseline parameter by averaging
    num = len(angles_range)
    baseline = 0
    for delta_time, angle in zip(delta_times, angles_range):
        angle_rad = angle/180*math.pi
        baseline += (scale*delta_time*wheel_vel/(angle_rad/2))
    baseline /= num
        # pass # TODO: replace with your code to compute the baseline parameter using scale, wheel_vel, and delta_time
    print("The baseline parameter is estimated as {:.6f} m.".format(baseline))

    return baseline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    args, _ = parser.parse_known_args()

    ppi = Alphabot(args.ip,args.port)

    # calibrate pibot scale and baseline
    dataDir = "{}/param/".format(os.getcwd())

    # print('Calibrating PiBot scale...\n')
    # scale = calibrateWheelRadius()
    # fileNameS = "{}scale.txt".format(dataDir)
    # np.savetxt(fileNameS, np.array([scale]), delimiter=',')
    scale = 1.291044301913867390e-02

    print('Calibrating PiBot baseline...\n')
    baseline = calibrateBaseline(scale)
    fileNameB = "{}baseline.txt".format(dataDir)
    np.savetxt(fileNameB, np.array([baseline]), delimiter=',')

    print('Finished calibration')
