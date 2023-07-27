# M4 - Autonomous fruit searching

# basic python packages
from util.measure import Drive
import sys
import os
import numpy as np
import json
import ast
import argparse
import time
import math
import random

import pygame # python package for GUI

from operate import Operate
from TargetPoseEst import FruitPoseEst
from AStar import AStar

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))

# import utility functions
sys.path.insert(0, "util")

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search
    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(
                        fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits
    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order
    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(
                                                      fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1

def generate_mid_waypoint(waypoint, robot_pose, small_dist):
    dist = math.hypot(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
    mid_waypoint = waypoint.copy()
    
    if dist < small_dist:
        return mid_waypoint
    
    delta_pose = math.atan2(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
    mid_waypoint[0] = robot_pose[0] + small_dist*np.cos(delta_pose)
    mid_waypoint[1] = robot_pose[1] + small_dist*np.sin(delta_pose)
    
    return mid_waypoint

# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    
    # Moving the robot one small dist at a time, while using slam in between
    # Adjustible parameters
    # TODO: Modify these and see what works
    wheel_vel = 20  # tick to move the robot, higher means faster, and batteries run out faster
    small_dist = 0.4 # moving a short distance multiple time to reach a longer distance
    dist_tolerance = 0.05 # tolerance to define whether the point is reached
    ####################################################
    
    moving_type = 1.2 # change moving type here

    dist = math.hypot(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
    first = True
    robot_pose = get_robot_pose()
    while dist > dist_tolerance:
        mid_waypoint = generate_mid_waypoint(waypoint, robot_pose, small_dist)
        
        if moving_type == 1.0:
            # Type 1.0: Moving after turning to mid_waypoint
            # turn towards the mid-waypoint
            pose_to_rot = math.atan2(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0]) - robot_pose[2]
            pose_to_rot = clamp_angle(pose_to_rot)
            
            anticlockwise = pose_to_rot > 0
            pose_to_rot = abs(pose_to_rot)
            
            # replace with your calculation
            turn_time = pose_to_rot/2*baseline/(scale*wheel_vel)
            print("Turning for {:.2f} seconds for {:.2f}".format(turn_time, pose_to_rot))
            move_and_update(wheel_vel = wheel_vel, dt = turn_time, turn_drive = 0, anticlockwise = anticlockwise)

            # after turning, drive straight to the waypoint
            mid_dist = math.hypot(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0])
            drive_time = mid_dist/(scale*wheel_vel)  # replace with your calculation
            print("Driving for {:.2f} seconds for {:.2f}m".format(drive_time, mid_dist))
            move_and_update(wheel_vel = wheel_vel, dt = drive_time, turn_drive = 1)
        
        elif moving_type == 1.1:
            # Type 1.1: Moving after turning straight to waypoint
            # turn towards the waypoint
            pose_to_rot = math.atan2(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0]) - robot_pose[2]
            pose_to_rot = clamp_angle(pose_to_rot)
            
            anticlockwise = pose_to_rot > 0
            pose_to_rot = abs(pose_to_rot)
            
            # replace with your calculation
            turn_time = (pose_to_rot/2)*(baseline/(scale*wheel_vel))
            print("Turning for {:.2f} seconds for {:.2f} rad".format(turn_time, pose_to_rot))
            move_and_update(wheel_vel = wheel_vel, dt = turn_time, turn_drive = 0, anticlockwise = anticlockwise)

            # after turning, drive straight to the waypoint
            dist = math.hypot(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
            drive_time = dist/(scale*wheel_vel)  # replace with your calculation
            print("Driving for {:.2f} seconds for {:.2f}m".format(drive_time, dist))
            move_and_update(wheel_vel = wheel_vel, dt = drive_time, turn_drive = 1)
        
        elif moving_type == 1.2:
            # Type 1.2: Moving after turning to mid_waypoint
            # segmented rotation
            
            # turn towards the waypoint with segmented rotation
            pose_to_rot = math.atan2(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0]) - robot_pose[2]
            pose_to_rot = clamp_angle(pose_to_rot)
            
            if abs(pose_to_rot) > 0.785: # if more than 45°
                turn_div = int(abs(pose_to_rot//0.785)) # only turn about 45° at a time
            else:
                turn_div = 1
            
            for i in reversed(range(turn_div+1)):
                pose_to_rot = math.atan2(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0]) - robot_pose[2]
                pose_to_rot = clamp_angle(pose_to_rot)
                
                if i != 0:
                    pose_to_rot = pose_to_rot/i
                
                anticlockwise = pose_to_rot > 0
                pose_to_rot = abs(pose_to_rot)
                
                # replace with your calculation
                turn_time = pose_to_rot/2*baseline/(scale*wheel_vel)
                print("Turning for {:.2f} seconds for {:.2f}rad".format(turn_time, pose_to_rot))
                move_and_update(wheel_vel = wheel_vel, dt = turn_time, turn_drive = 0, anticlockwise = anticlockwise)

                robot_pose = get_robot_pose() # update robot pose
            
            # after turning, drive straight to the waypoint
            mid_waypoint = generate_mid_waypoint(waypoint, robot_pose, small_dist)
            mid_dist = math.hypot(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0])
            drive_time = mid_dist/(scale*wheel_vel)  # replace with your calculation
            print("Driving for {:.2f} seconds for {:.2f}m".format(drive_time, mid_dist))
            move_and_update(wheel_vel = wheel_vel, dt = drive_time, turn_drive = 1)
        
            # use the angularly closest aruco for position estimation
            aruco_localisation()
            
        elif moving_type == 2.0:
            # Type 2: Moving while turning with Proportional controller
            
            # Parameters
            Kp_w = 30
            Kp_v = 500
            Ki_w = 1
            Ki_v = 1
            Kd_w = 1
            Kd_v = 1
            dt = 0.5 # interval between movement and taking pictures
            
            if first: # initialisation
                # initialising variables for PID controller
                cumError_v = 0
                rateError_v = 0
                cumError_w = 0
                rateError_w = 0   
                first = False

            pose_to_rot = math.atan2(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0]) - robot_pose[2]
            pose_to_rot = clamp_angle(pose_to_rot)
            
            mid_dist = math.hypot(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0])
            
            cumError_v += mid_dist*dt
            cumError_w += pose_to_rot*dt
            
            rateError_v = (mid_dist - rateError_v)/dt
            rateError_w = (pose_to_rot - rateError_w)/dt 
            
            max_dist = small_dist # maximum of mid_waypoint in the arena
            max_pose = math.pi
            lin_wheel_vel = Kp_v*normalise(mid_dist, min_val = 0, max_val = max_dist) + Ki_v*cumError_v + Kd_v*rateError_v
            rot_wheel_vel = Kp_w*normalise(pose_to_rot, min_val = 0, max_val = max_pose) + Ki_w*cumError_w + Kd_w*rateError_w
             
            # clamping
            MAX_TICK = 30
            MIN_TICK = 0
            
            lin_wheel_vel = clamp(lin_wheel_vel, MAX_TICK, MIN_TICK)
            rot_wheel_vel = clamp(abs(rot_wheel_vel), MAX_TICK, MIN_TICK) if rot_wheel_vel > 0 else -clamp(abs(rot_wheel_vel), MAX_TICK, MIN_TICK)
            
            print("Currently at at [{}, {}], dest [{}, {}]".format(robot_pose[0], robot_pose[1], mid_waypoint[0], mid_waypoint[1]))
            print('Pose to rot: {}, rot_wheel_vel = {}'.format(pose_to_rot, rot_wheel_vel))
            print('Mid dist: {}, lin_wheel_vel = {}\n'.format(mid_dist, lin_wheel_vel))
            
            move_and_update(wheel_vel=wheel_vel, dt=dt, turn_drive=2, anticlockwise=rot_wheel_vel>0, lin_wheel_vel=lin_wheel_vel, rot_wheel_vel=rot_wheel_vel)

        elif moving_type == 2.2:
            # Type 2.2: Moving while turning with Proportional controller
            
            # Parameters
            Kp_w = 30
            Kp_v = 500
            Ki_w = 1
            Ki_v = 1
            Kd_w = 1
            Kd_v = 1
            dt = 0.1 # interval between movement and taking pictures
            
            if first: # initialisation
                # initialising variables for PID controller
                cumError_v = 0
                rateError_v = 0
                cumError_w = 0
                rateError_w = 0   
                first = False
            
            #separating moving and turning
            pose_to_rot = math.atan2(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0]) - robot_pose[2]
            pose_to_rot = clamp_angle(pose_to_rot)
            
            mid_dist = math.hypot(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0])
            
            cumError_v += mid_dist*dt
            cumError_w += pose_to_rot*dt
            
            rateError_v = (mid_dist - rateError_v)/dt
            rateError_w = (pose_to_rot - rateError_w)/dt 
            
            max_dist = small_dist # maximum of mid_waypoint in the arena
            max_pose = math.pi
            lin_wheel_vel = Kp_v*normalise(mid_dist, min_val = 0, max_val = max_dist) + Ki_v*cumError_v + Kd_v*rateError_v
            rot_wheel_vel = Kp_w*normalise(pose_to_rot, min_val = 0, max_val = max_pose) + Ki_w*cumError_w + Kd_w*rateError_w
             
            # clamping
            MAX_TICK = 30
            MIN_TICK = 0
            
            lin_wheel_vel = clamp(lin_wheel_vel, MAX_TICK, MIN_TICK)
            rot_wheel_vel = clamp(abs(rot_wheel_vel), MAX_TICK, MIN_TICK) if rot_wheel_vel > 0 else -clamp(abs(rot_wheel_vel), MAX_TICK, MIN_TICK)
            
            print("Currently at at [{}, {}], dest [{}, {}]".format(robot_pose[0], robot_pose[1], mid_waypoint[0], mid_waypoint[1]))
            print('Pose to rot: {}, rot_wheel_vel = {}'.format(pose_to_rot, rot_wheel_vel))
            print('Mid dist: {}, lin_wheel_vel = {}\n'.format(mid_dist, lin_wheel_vel))
            
            anticlockwise = rot_wheel_vel > 0
            rot_wheel_vel = abs(rot_wheel_vel)

            move_and_update(wheel_vel = rot_wheel_vel, dt = dt, turn_drive = 0, anticlockwise = anticlockwise)
            move_and_update(wheel_vel = lin_wheel_vel, dt = dt, turn_drive = 1)
        else: 
            print('Invalid moving type')
        
        
        robot_pose = get_robot_pose()
        dist = math.hypot(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
        if moving_type == 1.1: # only move once without correction
            dist = dist_tolerance
        print("Currently at at [{}, {}, {}]".format(robot_pose[0], robot_pose[1], robot_pose[2]))
        ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    all_states = operate.ekf.get_state_vector().reshape(1, -1)
    robot_pose = list([all_states[0][0], all_states[0][1], all_states[0][2]])  # replace with your calculation
    ####################################################

    return robot_pose

def clamp(val, max_val, min_val):
    new_val = max(val, min_val)
    new_val = min(new_val, max_val)
    return new_val

def clamp_angle(rad_angle=0, min_value=-np.pi, max_value=np.pi):
	"""
	Restrict angle to the range [min, max]
	:param rad_angle: angle in radians
	:param min_value: min angle value
	:param max_value: max angle value
	"""

	if min_value > 0:
		min_value *= -1

	angle = (rad_angle + max_value) % (2 * np.pi) + min_value

	return angle

def normalise(val, min_val, max_val):
    
    return (val-min_val)/(max_val-min_val)
  
def move_and_update(wheel_vel=10, dt = 1e-3, turn_drive = 0, anticlockwise = True, lin_wheel_vel = 10, rot_wheel_vel = 10, force_est = False):
    pygame.event.get() # prevent pygame window from crashing
    if dt == 0: # skip moving if no motion is needed
        lv = 0
        rv = 0
    else:
        if turn_drive == 0: # turning only
            if anticlockwise:
                move = [0, 1]
            else:
                move = [0, -1]

            lv, rv = operate.pibot.set_velocity(move, turning_tick=wheel_vel, time=dt)
        elif turn_drive == 1: # translating only
            move = [1, 0]
            lv, rv = operate.pibot.set_velocity(move, tick=wheel_vel, time=dt)
        elif turn_drive == 2: # turning and moving
            if anticlockwise:
                move = [1, 1]
            else:
                move = [1, -1]
            rot_wheel_vel = abs(rot_wheel_vel)

            print("Driving for {:.2f} seconds with v = {:.2f}, w = {:.2f}".format(dt, lin_wheel_vel, rot_wheel_vel))
            lv, rv = operate.pibot.set_velocity(move, tick=lin_wheel_vel, turning_tick=rot_wheel_vel, time=dt)
        else:
            print('Warning: Unidentified move command')
    
    #time.sleep(0.2) # wait for a stable image [Delay is already put into operate.take_pic]
    operate.take_pic()
    drive_meas = Drive(lv, rv, dt)
    lms = operate.update_slam(drive_meas)

    # running detector and saving detector
    operate.command['inference'] = True
    
    # do not save raw images during proper run
    if args.save_img == 1:
        operate.command['save_image'] = True 

    operate.save_image()
    num_targets, detected_targets = operate.detect_target()
    
    for target in detected_targets:
        if target in fruitPoseEst.unconfirmed_fruits_idx_list:
            operate.new_targets_found = True
    
    if num_targets > 0: # only save inference if something is detected
        operate.command['save_inference'] = True
    
    operate.record_data()

    if num_targets > 0 or force_est: # only recalculate if something is detected
        fruitPoseEst.calc_est_pose()
        
    # visualise
    update_display()
    
    return lms

# spin 360° to localise the robot
def spin_localisation(div = 8, quick_spin = False):
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    robot_pose = get_robot_pose()
    total_rotation = 2*math.pi + robot_pose[2]
    
    wheel_vel = 20
    
    if quick_spin:
        operate.notification = 'Aruco-spin-localising...'
    else:
        operate.notification = 'Spin-localising...'

    for d in reversed(range(div+1)):
        robot_pose = get_robot_pose()
        if d != 0:
            pose_to_rot = (total_rotation - robot_pose[2])/d
        else: 
            pose_to_rot = (total_rotation - robot_pose[2])
        pose_to_rot = clamp_angle(pose_to_rot)
        anticlockwise = pose_to_rot > 0
        pose_to_rot = abs(pose_to_rot)
        turn_time = pose_to_rot/2*baseline/(scale*wheel_vel)
        if quick_spin:
            print("Aruco-spin localising: Turning for {:.4f} seconds for {:.4f} rad, {}".format(turn_time, pose_to_rot, d))
        else:
            print("Spin localising: Turning for {:.4f} seconds for {:.4f} rad".format(turn_time, pose_to_rot))
        lms = move_and_update(wheel_vel = wheel_vel, dt = turn_time, turn_drive = 0, anticlockwise = anticlockwise)
        
        if quick_spin: # quick spin mode: stop immediately when an aruco marker is detected
            if len(lms) > 0:
                return lms
            
    return lms

def get_nearest_aruco():
    robot_pose = get_robot_pose()
    
    # TODO: calculate the nearest aruco markers
    angle_diff = np.zeros(10)
    for idx, aruco_pos in enumerate(aruco_true_pos):
        angle_diff[idx] = (clamp_angle(math.atan2(aruco_pos[1] - robot_pose[1], aruco_pos[0] - robot_pose[0]) - robot_pose[2]))
        
    order = np.argsort(abs(angle_diff))
    angle_diff_sorted = angle_diff[order]
    return angle_diff_sorted

def aruco_localisation():
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    wheel_vel = 20
    
    operate.notification = 'Aruco-localising...'
    
    aruco_pos_list = get_nearest_aruco()
    
    # turn to the closest only
    pose_to_rot = aruco_pos_list[0].item()
    pose_to_rot = clamp_angle(pose_to_rot)
    anticlockwise = pose_to_rot > 0
    pose_to_rot = abs(pose_to_rot)
    turn_time = pose_to_rot/2*baseline/(scale*wheel_vel)
    print("Aruco-localising: Turning for {:.2f} seconds for {:.2f} rad".format(turn_time, pose_to_rot))
    lms = move_and_update(wheel_vel = wheel_vel, dt = turn_time, turn_drive = 0, anticlockwise = anticlockwise)

    while len(lms) == 0: 
        lms = spin_localisation(div = 12, quick_spin = True)

def drive_forward(dist): 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    
    wheel_vel = 20
    # after turning, drive straight to the waypoint
    drive_time = dist/(scale*wheel_vel)  # replace with your calculation
    print("Driving for {:.2f} seconds for {:.2f}m".format(drive_time, dist))
    move_and_update(wheel_vel = wheel_vel, dt = drive_time, turn_drive = 1)
    
def to_m(uv, res = (500, 500), offset = (0, 0), m2pixel = 156.25):
    w, h = res
    u, v = uv[0] - offset[0], uv[1] - offset[1]
    x = (u-w/2)/m2pixel
    y = -(v-h/2)/m2pixel
    return (x, y)

def to_im_coor(xy, res, m2pixel):
    w, h = res
    x, y = xy
    x_im = int(x*m2pixel+w/2.0)
    y_im = int(-y*m2pixel+h/2.0)
    return (x_im, y_im)

def create_waypoints_surface(waypoints_list, waypoint_pics):
    LABEL_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 15)
    waypoints_surface = pygame.Surface([500, 500], pygame.SRCALPHA, 32) # transparent surface
    for index, xy in enumerate(waypoints_list):
        x_im, y_im = to_im_coor(xy, [500, 500], m2pixel=156.25)
        waypoints_surface.blit(waypoint_pics[index%len(waypoint_pics)], (x_im-10, y_im-20))
        label = LABEL_FONT.render(f'{index+1}', False, (0, 0, 0))
        waypoints_surface.blit(label, (x_im-label.get_width()/4, y_im-20))
    return waypoints_surface

def update_display():
    # draw base map and SLAM and robot pose
    operate.draw(canvas)
    
    # drawing path
    if args.level == 2 and len(a_star_waypoints) != 0:
        path_surface = create_path_surface(a_star_waypoints, res, res_a_star)
        canvas.blit(path_surface, map_location)
    
    # drawing waypoints
    waypoints_surface = create_waypoints_surface(waypoints_list, waypoint_pics)
    canvas.blit(waypoints_surface, map_location)
    
    # drawing target fruits
    fruits_surface = fruitPoseEst.draw(res = res, m2pixel = m2pixel)
    canvas.blit(fruits_surface, map_location)
    
    pygame.display.update()

def create_obstacle_map(astar_obstacle_file_path, res, sz = (3.2, 3.2)):
    m2pixel = res[1]/sz[1]
    
    map = 255*np.ones(res).astype(np.uint8)

    # borders
    border_width = 0.1
    corner_cube = to_im_coor((-sz[0]/2+border_width, sz[1]/2-border_width), res, m2pixel)
    top_left_coor = to_im_coor((-sz[0]/2, sz[1]/2), res, m2pixel)
    top_right_coor = to_im_coor((sz[0]/2, sz[1]/2), res, m2pixel)
    bottom_left_coor = to_im_coor((-sz[0]/2, -sz[1]/2), res, m2pixel)
    bottom_right_coor = to_im_coor((sz[0]/2, -sz[1]/2), res, m2pixel)
    
    map[top_left_coor[1]:bottom_left_coor[1],top_left_coor[0]:(top_left_coor[0] + corner_cube[0])] = 0 # left border 
    map[top_right_coor[1]:bottom_right_coor[1],(top_right_coor[0] - corner_cube[0]):top_right_coor[0]] = 0 # right border 
    map[top_left_coor[1]:(top_left_coor[1] + corner_cube[1]),top_left_coor[0]:top_right_coor[0]] = 0 # top border 
    map[(bottom_right_coor[1] - corner_cube[1]):bottom_right_coor[1],bottom_left_coor[0]:bottom_right_coor[0]] = 0 # bottom border

    object_clearance = 0.15
    # aruco markers
    markers = operate.ekf.markers
    cube = to_im_coor((-sz[0]/2+object_clearance, sz[1]/2-object_clearance), res, m2pixel)
    
    for i in range(markers.shape[1]):
        x = markers[0][i]
        y = markers[1][i]
        
        xy_coor = to_im_coor((x, y), res, m2pixel)
        
        # draw a square
        map[(xy_coor[1] - cube[1]):(xy_coor[1] + cube[1]), (xy_coor[0] - cube[0]):(xy_coor[0] + cube[0])] = 0
        
    # fruits
    fruits_locations = fruitPoseEst.get_search_order()
    
    for [x, y] in fruits_locations:
        xy_coor = to_im_coor((x, y), res, m2pixel)
        
        # draw a square
        map[(xy_coor[1] - cube[1]):(xy_coor[1] + cube[1]), (xy_coor[0] - cube[0]):(xy_coor[0] + cube[0])] = 0
    
    data = []
    idx_y, idx_x = np.where(map == 0)
    
    for y, x in zip(idx_y, idx_x):
        data.append([int(x), int(y)])
    
    write_dict = {}    
    write_dict["data"] = data
    
    with open(astar_obstacle_file_path, "w") as outfile:
        json_data = json.dumps(write_dict)
        outfile.write(json_data)
        
    return map

def initialiseAStar(json_file_path, res, start, end):
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    data = contents["data"]
    obstacle_ratio = 10 
    a_star = AStar(res[0],  res[1], [start[0], start[1]], [end[0], end[1]], obstacle_ratio, data)
    return a_star

def find_closest_at_target(target_pose, robot_pose, clearance = 0.2121):
    mid_target = target_pose.copy()
    delta_pose = math.atan2(target_pose[1] - robot_pose[1], target_pose[0] - robot_pose[0])
    mid_target[0] = target_pose[0] - clearance*np.cos(delta_pose)
    mid_target[1] = target_pose[1] - clearance*np.sin(delta_pose)
    init_delta_pose = delta_pose
    while not check_no_obstruction(mid_target):
        delta_pose += 0.08727 # increase delta_pose by 5°
        mid_target[0] = target_pose[0] - clearance*np.cos(delta_pose)
        mid_target[1] = target_pose[1] - clearance*np.sin(delta_pose)
        
        if abs(delta_pose) >= init_delta_pose + 2*np.pi: # give up searching if a circle around it is all obstructed
            print('Cannot find unobstructed closest point')
            break
    
    dist = math.hypot(target_pose[1] - robot_pose[1], target_pose[0] - robot_pose[0])
    if dist < clearance:
        mid_target[0] = robot_pose[0]
        mid_target[1] = robot_pose[1]

    return mid_target

def check_no_obstruction(mid_target):
    target_x, target_y = to_im_coor((mid_target[0], mid_target[1]), res_a_star, m2pixel=res_a_star[0]/sz[0])
    
    if obstacle_map[target_y, target_x] == 0:
        return False
    
    return True
    

def create_path_surface(a_star_waypoints, res, res_a_star):
    path_surface = pygame.Surface([500, 500], pygame.SRCALPHA, 32) # transparent surface
    if len(a_star_waypoints) == 0:
        return 
    
    # color = (218, 142, 231) # path in violet
    for xy in a_star_waypoints:
        xy_projected = (int(1.0*xy.x/res_a_star[0]*res[0]), int(1.0*xy.y/res_a_star[1]*res[1]))
        pellet_img = pygame.image.load('pics/8bit/pellet.png')
        path_surface.blit(pellet_img, (xy_projected[0] - pellet_img.get_width()//2, xy_projected[1] - pellet_img.get_height()//2))
        # path_surface.set_at(xy_projected, color)
    
    return path_surface

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    parser.add_argument("--level", metavar='', type=int, default=2)
    parser.add_argument("--save_img", metavar='', type=int, default=0)
    parser.add_argument("--return_origin", metavar='', type=int, default=0)
    parser.add_argument("--waypoint_inc", metavar='', type=int, default=6)
    args, _ = parser.parse_known_args()

    # GUI
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    text_colour = (220, 220, 220)

    width, height = 880, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2022 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading_large.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    waypoint_pics = []
    for i in range(1, 11):
        f_ = f'./pics/8bit/waypoint ({i}).png'
        waypoint_pics.append(pygame.image.load(f_))

    counter = 40
    fire_count = 1

    # Start Screen
    start = False
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        
        x_ = min(counter, 800)
        
        if x_ < 800:
            canvas.blit(splash, (0, 0))
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2
        else:
            # crashing animation -- don't ask why
            if fire_count == 1:
                canvas.blit(pygame.image.load('pics/8bit/explosion2.png'), (x_-100, 565-100))
            else:
                if fire_count%10 == 0:
                    rand_x = random.randint(int(800-fire_count/10), width-25)
                    rand_y = random.randint(int(565-fire_count/10), height-25)
                    canvas.blit(pygame.image.load('pics/8bit/fire.png'), (rand_x, rand_y))
            pygame.display.update()
            fire_count += 1
    
    canvas.blit(splash, (0, 0))

    operate = Operate(args)
    operate.ekf_on = True # turn ekf on
    
    operate.draw(canvas)
    pygame.display.update()

    v_pad = 40
    h_pad = 20
    sz = (3.2, 3.2)
    res = (500, 500)
    m2pixel = res[1]/sz[1] # 500 pixels / 3.2m
    map_location = (2*h_pad+320, v_pad)
    
    map_area = pygame.Surface(res)
    map_area.set_alpha(0)
    canvas.blit(map_area, map_location)
    
    map_rect = map_area.get_rect().move(map_location[0], map_location[1])
    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    operate.ekf.populate_landmarks(aruco_true_pos)
    
    fruitPoseEst = FruitPoseEst(fruits_list, fruits_true_pos, search_list)
    fruitPoseEst.calc_est_pose()
    
    astar_obstacle_file_path = "astar_obstacles.json"
    res_a_star = (100, 100)
    obstacle_map = create_obstacle_map(astar_obstacle_file_path, res_a_star)
    a_star_waypoints = []
    
    waypoint = [0.0, 0.0]
    robot_pose = [0.0, 0.0, 0.0]
    waypoints_list = []
    waypoint_idx_increment = args.waypoint_inc
    
    # update once before drawing
    move_and_update(wheel_vel=0, dt = 1e-3, turn_drive = 0, force_est = True)
    
    # drawing annotations
    update_display()

    waypoint_idx = 0
    target_idx = 0
    targets_list = []
    targets_list = fruitPoseEst.get_search_order() # initialise targets_list
    
    # calculate origin location in A* map resolution
    origin_x, origin_y = to_im_coor((0.0, 0.0), res_a_star, m2pixel=res_a_star[1]/sz[1])
    
    # Mandatory spin at the start
    spin_localisation(div=8)

    stop_flag = False
    reached = False
    while True:
        x,y = 0.0,0.0
        
        # enter the waypoints
        # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
        if args.level == 1: # level 1
            operate.notification = f'Select waypoint {len(waypoints_list)+1} by clicking on map'
            update_display()

            # Detect waypoint from mouse input
            clicked = False
            while not clicked:
                events = pygame.event.get() # must call to keep window active
                # move_and_update(wheel_vel=0, dt = 1e-3, turn_drive = 0) # scanning while idle
                operate.notification = f'Select waypoint {len(waypoints_list)+1} by clicking on map'
                update_display()
                for event in events:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        spin_localisation(div=8)
                    
                    if map_rect.collidepoint(pygame.mouse.get_pos()):
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
                        xy = to_m(pygame.mouse.get_pos(), res = res, offset=map_location, m2pixel=m2pixel)
                        mouse_xy_text = f'{xy[0]:3.3f}, {xy[1]:3.3f}'
                        mouse_xy_surface = TEXT_FONT.render(mouse_xy_text, False, (50, 50, 50), (255, 255, 255))
                        mouse_xy_bg = pygame.Surface((220+h_pad, mouse_xy_surface.get_height()))
                        canvas.blit(mouse_xy_bg, (width-h_pad-220, v_pad/2))
                        canvas.blit(mouse_xy_surface, (width-h_pad-220, v_pad/2))
                        pygame.display.update()
                        
                        if event.type == pygame.MOUSEBUTTONDOWN:
                                x = xy[0]
                                y = xy[1]
                                clicked = True
                                waypoint_idx += 1
                    else:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
            waypoints_list.append([x, y])
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
            
        elif args.level == 2 or args.level == 3: # level 2 and 3
            # Automate waypoints searching
            nav_type = 1 # select navigation type here
            
            reached = False
            # checking if the target is reached before navigating
            if len(targets_list) !=0:
                try:
                    target_pose = targets_list[target_idx]
                except:
                    stop_flag = True
                    target_idx = target_idx - 1
                    target_pose = targets_list[target_idx]
                    print('No more detected targets') 
                dist = math.hypot(target_pose[1] - robot_pose[1], target_pose[0] - robot_pose[0])
                if dist < 0.4:
                    reached = True
            
            if nav_type == 1:
                # Plan once only, will not react if an obstacle is newly discovered
                if operate.new_targets_found: # reset paths if new targets are found
                    waypoint_idx = 0
                    waypoints_list = [] # clear waypoints_list
                    # operate.new_targets_found = False 
                    print('Unknown targets found. Re-running path finding. ')

                if waypoint_idx == 0: # first waypoint 
                    if not operate.new_targets_found: 
                        spin_localisation(div=8) # spin and look for additional fruits
                    else:
                        operate.new_targets_found = False # do not spin again if new fruits are found
                    
                    targets_list = fruitPoseEst.get_search_order()
                    try:
                        target_pose = targets_list[target_idx]
                    except:
                        stop_flag = True
                        target_idx = target_idx - 1
                        target_pose = targets_list[target_idx]
                        print('No more detected targets') 
                    # path planning
                    mid_target = find_closest_at_target(target_pose, robot_pose, clearance = 0.2121)
                    robot_x, robot_y = to_im_coor((robot_pose[0], robot_pose[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                    mid_target_x, mid_target_y = to_im_coor((mid_target[0], mid_target[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                    
                    a_star_waypoints = []
                    first_path = True
                    while len(a_star_waypoints) == 0:
                        if not first_path: # preventing unecessary spins
                            spin_localisation(div = 8) # keep localising until a path is found
                            print('Spinning because path cannot be found')
    
                        operate.notification = 'Finding path...'
                        update_display()
                        
                        obstacle_map = create_obstacle_map(astar_obstacle_file_path, res_a_star)
                        a_star = initialiseAStar(astar_obstacle_file_path, res_a_star, start = [robot_x, robot_y], end = [mid_target_x, mid_target_y])
                        a_star_waypoints = a_star.main()
                        a_star_waypoints.reverse() # the list provided by astar is reversed   
                        first_path = False                     

                update_display()
                
                waypoint_idx += waypoint_idx_increment
                waypoint_idx = clamp(waypoint_idx, len(a_star_waypoints)-1, 0)

                xy = to_m((a_star_waypoints[waypoint_idx].x, a_star_waypoints[waypoint_idx].y), res_a_star, offset = (0, 0), m2pixel=res_a_star[1]/sz[1])
                x = xy[0]
                y = xy[1]
                
                if stop_flag:
                    x = robot_pose[0]
                    y = robot_pose[1]
                
                waypoints_list.append([x, y])
                
                if waypoint_idx >= len(a_star_waypoints)-1: # if target is reached
                    reached = True
                
                operate.notification = f'Moving to target {target_idx} waypoint {waypoint_idx} at {x}, {y}'
            
            elif nav_type == 2:
                # continuous path planning
                try:
                    targets_list = fruitPoseEst.get_search_order()
                    target_pose = targets_list[target_idx]
                except:
                    stop_flag = True
                    target_idx = target_idx - 1
                    print('No more detected targets') 
                
                mid_target = find_closest_at_target(target_pose, robot_pose, clearance = 0.1414)
                robot_x, robot_y = to_im_coor((robot_pose[0], robot_pose[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                mid_target_x, mid_target_y = to_im_coor((mid_target[0], mid_target[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                
                operate.notification = 'Finding path...'
                update_display()
                
                obstacle_map = create_obstacle_map(astar_obstacle_file_path, res_a_star)
                a_star = initialiseAStar(astar_obstacle_file_path, res_a_star, start = [robot_x, robot_y], end = [mid_target_x, mid_target_y])
                a_star_waypoints = a_star.main()
                a_star_waypoints.reverse() # the list provided by astar is reversed

                update_display() # display path
                
                skip_idx = waypoint_idx_increment # project forward since each step only move the robot by 1 pixel, or 3.2mm x 3 is about 0.1m
                if skip_idx > len(a_star_waypoints)-1:
                    reached = True
                    skip_idx = len(a_star_waypoints) - 1
                    
                if len(a_star_waypoints) == 0: # stay at current position
                    reached = True
                    x = robot_pose[0]
                    y = robot_pose[1]
                else:
                    xy = to_m((a_star_waypoints[skip_idx].x, a_star_waypoints[skip_idx].y), res_a_star, offset = (0, 0), m2pixel=res_a_star[1]/sz[1])
                    x = xy[0]
                    y = xy[1]
                
                waypoints_list.append([x, y])
                waypoint_idx += 1                
                
                operate.notification = f'Moving to target {target_idx} waypoint {waypoint_idx} at {x}, {y}'
                
        
        elif args.level == -99: # debugging only
            div = 8
            spin_localisation(div=div)
            while True:
                operate.notification = 'Press space to spin'
                update_display()
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        spin_localisation(div=div)
                
            # while True: 
            #     operate.notification = 'Press space to move forward'
            #     update_display()
            #     events = pygame.event.get()
            #     for event in events:
            #         if event.type == pygame.QUIT:
            #             pygame.quit()
            #             sys.exit()
            #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            #             pygame.quit()
            #             sys.exit()
            #         elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            #             drive_forward(0.4)
                
        update_display()

        # estimate the robot's pose
        robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint, robot_pose)
        robot_pose = get_robot_pose()
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        move_and_update(wheel_vel=0, dt = 1e-3, turn_drive = 0) # stop the robot
    
        if reached: # if target is reached
            target_idx += 1 # next target
            waypoint_idx = 0
            waypoints_list = [] # clear waypoints_list
            print(f'Reached target {target_idx}, pausing for 5 seconds')
            for i in reversed(range(5)): # stay in place for 8 seconds
                if i == 0:
                    operate.notification = f'Reached target {target_idx}, resuming in {i+1} second'
                else:
                    operate.notification = f'Reached target {target_idx}, resuming in {i+1} seconds'
                pygame.event.get() # preventing the window from crashing
                update_display()
                pygame.time.wait(1000) 

        if args.level != 1 and (target_idx >= 3 or stop_flag): #(target_idx >= 5 or stop_flag):
            while True:
                operate.notification = 'Done navigating. (Visited all 3 targets)' #Done navigating. (Visited all 3 targets)'
                print(f'Done navigating. (Visited all 3 targets)')

                if stop_flag:
                    operate.notification = 'Done navigating. (No more target detected)'
                    print('Done navigating. (No more target detected)')
                update_display()
                events = pygame.event.get() # prevent window from crashing
                for event in events:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                
        if reached:
            if args.return_origin == 1:
                # go back to origin
                robot_x, robot_y = to_im_coor((robot_pose[0], robot_pose[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                
                a_star_waypoints = []
                
                while len(a_star_waypoints) == 0:
                    spin_localisation(div = 8)
                    # return to origin
                    operate.notification = 'Finding path back to origin...'
                    update_display()
                    obstacle_map = create_obstacle_map(astar_obstacle_file_path, res_a_star)
                    a_star = initialiseAStar(astar_obstacle_file_path, res, start = [robot_x, robot_y], end = [origin_x, origin_y])
                    a_star_waypoints = a_star.main()
                    a_star_waypoints.reverse()
                    
                    if len(a_star_waypoints) == 0:
                        print('Fail to find path')
                
                waypoint_idx = 0
                
                origin_reached = False
                while not origin_reached:
                    operate.notification = 'Navigating back to origin'
                    update_display()
                    
                    if operate.new_targets_found: # reset paths if new targets are found
                        waypoint_idx = 0
                        waypoints_list = [] # clear waypoints_list
                        print('Unknown targets found. Re-running path finding. ')
                    
                    first_path = True
                    while len(a_star_waypoints) == 0:
                        if not first_path:
                            if not operate.new_targets_found:
                                spin_localisation(div = 8)
                            else:
                                operate.new_targets_found = False
                            
                        # return to origin
                        operate.notification = 'Finding path back to origin...'
                        update_display()
                        obstacle_map = create_obstacle_map(astar_obstacle_file_path, res_a_star)
                        a_star = initialiseAStar(astar_obstacle_file_path, res, start = [robot_x, robot_y], end = [origin_x, origin_y])
                        a_star_waypoints = a_star.main()
                        a_star_waypoints.reverse()
                        
                        first_path = False
                        if len(a_star_waypoints) == 0:
                            print('Fail to find path')
                    
                    waypoint_idx += waypoint_idx_increment
                    if waypoint_idx >= len(a_star_waypoints)-1:
                        waypoint_idx = len(a_star_waypoints)-1
                        origin_reached = True
                    
                    xy = to_m((a_star_waypoints[waypoint_idx].x, a_star_waypoints[waypoint_idx].y), res_a_star, offset = (0, 0), m2pixel=res_a_star[1]/sz[1])
                    x = xy[0]
                    y = xy[1]
                    
                    waypoint = [x, y]
                    waypoints_list.append([x, y])
                    
                    drive_to_point(waypoint, robot_pose)
                    robot_pose = get_robot_pose()
                    
                    move_and_update(wheel_vel=0, dt = 1e-3, turn_drive = 0) # stop the robot
                    
                # reset waypoints
                waypoint_idx = 0
                waypoints_list = [] # clear waypoints_list