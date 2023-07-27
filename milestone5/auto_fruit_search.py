# M4 - Autonomous fruit searching

# basic python packages
from util.measure import Drive
import sys
import os
import numpy as np
import json
import argparse
import math
import random
import time

import pygame # python package for GUI

from operate import Operate
from AStar import AStar

from util.helper import read_search_list, read_true_map, print_target_fruits_pos, to_im_coor, to_m, clamp, clamp_angle, normalise

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))

# import utility functions
sys.path.insert(0, "util")

pygame.font.init() 
TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
LABEL_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 15)
ALT_TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 43)

pellet_img = pygame.image.load('pics/8bit/pellet.png')

def print_toggleable(line):
    if args.verbose == 1:
        print(line)

def drive_to_point(waypoint, robot_pose):
    moving_type = 1.2 # change moving type here

    dist = math.hypot(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
    first = True
    robot_pose = operate.get_robot_pose()
    while dist > operate.dist_tolerance:
        mid_waypoint = operate.generate_mid_waypoint(waypoint, robot_pose)
        
        if moving_type == 1.0:
            # Type 1.0: Moving after turning to mid_waypoint
            # turn towards the mid-waypoint
            pose_to_rot = math.atan2(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0]) - robot_pose[2]
            pose_to_rot = clamp_angle(pose_to_rot)
            
            anticlockwise = np.sign(pose_to_rot)
            pose_to_rot = abs(pose_to_rot)
            
            # replace with your calculation
            turn_time = pose_to_rot/2*operate.baseline/(operate.scale*operate.wheel_vel)
            print_toggleable("Turning for {:.2f} seconds for {:.2f}".format(turn_time, pose_to_rot))
            move_and_update(wheel_vel = operate.wheel_vel, dt = turn_time, turn_drive = 0, dir = anticlockwise)

            # after turning, drive straight to the waypoint
            mid_dist = math.hypot(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0])
            drive_time = mid_dist/(operate.scale*operate.wheel_vel)  # replace with your calculation
            print_toggleable("Driving for {:.2f} seconds for {:.2f}m".format(drive_time, mid_dist))
            move_and_update(wheel_vel = operate.wheel_vel, dt = drive_time, turn_drive = 1)
        
        elif moving_type == 1.1:
            # Type 1.1: Moving after turning straight to waypoint
            # turn towards the waypoint
            pose_to_rot = math.atan2(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0]) - robot_pose[2]
            pose_to_rot = clamp_angle(pose_to_rot)
            
            anticlockwise = np.sign(pose_to_rot)
            pose_to_rot = abs(pose_to_rot)
            
            # replace with your calculation
            turn_time = (pose_to_rot/2)*(operate.baseline/(operate.scale*operate.wheel_vel))
            print_toggleable("Turning for {:.2f} seconds for {:.2f} rad".format(turn_time, anticlockwise*pose_to_rot))
            move_and_update(wheel_vel = operate.wheel_vel, dt = turn_time, turn_drive = 0, dir = anticlockwise)

            # after turning, drive straight to the waypoint
            dist = math.hypot(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
            drive_time = dist/(operate.scale*operate.wheel_vel)  # replace with your calculation
            print_toggleable("Driving for {:.2f} seconds for {:.2f}m".format(drive_time, dist))
            move_and_update(wheel_vel = operate.wheel_vel, dt = drive_time, turn_drive = 1)
        
        elif moving_type == 1.2:
            # Type 1.2: Moving after turning to mid_waypoint
            # segmented rotation
            
            # turn towards the waypoint with segmented rotation
            pose_to_rot = math.atan2(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0]) - robot_pose[2]
            pose_to_rot = clamp_angle(pose_to_rot)
            
            if abs(pose_to_rot) > 0.785: # if more than 45°
                turn_div = int(abs(pose_to_rot//0.785) + 1) # only turn about 45° at a time
            else:
                turn_div = 1
            
            for i in reversed(range(turn_div+1)):
                robot_pose = operate.get_robot_pose() # update robot pose
                pose_to_rot = math.atan2(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0]) - robot_pose[2]
                pose_to_rot = clamp_angle(pose_to_rot)
                
                if i != 0:
                    pose_to_rot = pose_to_rot/i
                
                anticlockwise = np.sign(pose_to_rot)
                pose_to_rot = abs(pose_to_rot)
                
                # replace with your calculation
                turn_time = pose_to_rot/2*operate.baseline/(operate.scale*operate.wheel_vel)
                print_toggleable("Turning for {:.2f} seconds for {:.2f}rad".format(turn_time, pose_to_rot))
                lms = move_and_update(wheel_vel = operate.wheel_vel, dt = turn_time, turn_drive = 0, dir = anticlockwise)

                robot_pose = operate.get_robot_pose() # update robot pose

                if args.ir_backoff == 1:
                    lms = ir_backoff(lms_prev = lms)
            
            robot_pose = operate.get_robot_pose() # update robot pose
            # after turning, drive straight to the waypoint
            mid_waypoint = operate.generate_mid_waypoint(waypoint, robot_pose)
            
            if not operate.reset_path:
                mid_dist = math.hypot(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0])
                drive_time = mid_dist/(operate.scale*operate.wheel_vel)  # replace with your calculation
                print_toggleable("Driving for {:.2f} seconds for {:.2f}m".format(drive_time, mid_dist))
                lms = move_and_update(wheel_vel = operate.wheel_vel, dt = drive_time, turn_drive = 1)
                time.sleep(0.3) # prevent drifting
            
            # use the angularly closest aruco for position estimation
            if args.aruco_localise == 1:
                aruco_localisation(curlms = lms)
                
            if operate.reset_path: # stop moving to the current waypoint if detected an obstacle
                break
    
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
            
            max_dist = operate.small_dist # maximum of mid_waypoint in the arena
            max_pose = math.pi
            lin_wheel_vel = Kp_v*normalise(mid_dist, min_val = 0, max_val = max_dist) + Ki_v*cumError_v + Kd_v*rateError_v
            rot_wheel_vel = Kp_w*normalise(pose_to_rot, min_val = 0, max_val = max_pose) + Ki_w*cumError_w + Kd_w*rateError_w
             
            # clamping
            MAX_TICK = 30
            MIN_TICK = 0
            
            lin_wheel_vel = clamp(lin_wheel_vel, MAX_TICK, MIN_TICK)
            rot_wheel_vel = clamp(abs(rot_wheel_vel), MAX_TICK, MIN_TICK) if rot_wheel_vel > 0 else -clamp(abs(rot_wheel_vel), MAX_TICK, MIN_TICK)
            
            print_toggleable("Currently at at [{}, {}], dest [{}, {}]".format(robot_pose[0], robot_pose[1], mid_waypoint[0], mid_waypoint[1]))
            print_toggleable('Pose to rot: {}, rot_wheel_vel = {}'.format(pose_to_rot, rot_wheel_vel))
            print_toggleable('Mid dist: {}, lin_wheel_vel = {}\n'.format(mid_dist, lin_wheel_vel))
            
            move_and_update(wheel_vel=operate.wheel_vel, dt=dt, turn_drive=2, dir=np.sign(rot_wheel_vel), lin_wheel_vel=lin_wheel_vel, rot_wheel_vel=rot_wheel_vel)

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
            
            max_dist = operate.small_dist # maximum of mid_waypoint in the arena
            max_pose = math.pi
            lin_wheel_vel = Kp_v*normalise(mid_dist, min_val = 0, max_val = max_dist) + Ki_v*cumError_v + Kd_v*rateError_v
            rot_wheel_vel = Kp_w*normalise(pose_to_rot, min_val = 0, max_val = max_pose) + Ki_w*cumError_w + Kd_w*rateError_w
             
            # clamping
            MAX_TICK = 30
            MIN_TICK = 0
            
            lin_wheel_vel = clamp(lin_wheel_vel, MAX_TICK, MIN_TICK)
            rot_wheel_vel = clamp(abs(rot_wheel_vel), MAX_TICK, MIN_TICK) if rot_wheel_vel > 0 else -clamp(abs(rot_wheel_vel), MAX_TICK, MIN_TICK)
            
            print_toggleable("Currently at at [{}, {}], dest [{}, {}]".format(robot_pose[0], robot_pose[1], mid_waypoint[0], mid_waypoint[1]))
            print_toggleable('Pose to rot: {}, rot_wheel_vel = {}'.format(pose_to_rot, rot_wheel_vel))
            print_toggleable('Mid dist: {}, lin_wheel_vel = {}\n'.format(mid_dist, lin_wheel_vel))
            
            anticlockwise = np.sign(rot_wheel_vel)
            rot_wheel_vel = abs(rot_wheel_vel)

            move_and_update(wheel_vel = rot_wheel_vel, dt = dt, turn_drive = 0, dir = anticlockwise)
            move_and_update(wheel_vel = lin_wheel_vel, dt = dt, turn_drive = 1)
        else: 
            print_toggleable('Invalid moving type')
        
        robot_pose = operate.get_robot_pose()
        dist = math.hypot(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
        if moving_type == 1.1: # only move once without correction
            dist = operate.dist_tolerance
        print_toggleable("Currently at at [{}, {}, {}]".format(robot_pose[0], robot_pose[1], robot_pose[2]))
        ####################################################

    print_toggleable("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
  
def move_and_update(wheel_vel=10, dt = 1e-3, turn_drive = 0, dir = 1, lin_wheel_vel = 10, rot_wheel_vel = 10, force_est = False):
    pygame.event.get() # prevent pygame window from crashing
    if dt == 0: # skip moving if no motion is needed
        lv = 0
        rv = 0
    else:
        if turn_drive == 0: # turning only
            if dir > 0:
                move = [0, 1]
            else:
                move = [0, -1]

            lv, rv = operate.pibot.set_velocity(move, turning_tick=wheel_vel, time=dt)
        elif turn_drive == 1: # translating only
            if dir > 0:
                move = [1, 0]
            else:
                move = [-1, 0]
            lv, rv = operate.pibot.set_velocity(move, tick=wheel_vel, time=dt)
        elif turn_drive == 2: # turning and moving
            if dir > 0:
                move = [1, 1]
            else:
                move = [1, -1]
            rot_wheel_vel = abs(rot_wheel_vel)

            print_toggleable("Driving for {:.2f} seconds with v = {:.2f}, w = {:.2f}".format(dt, lin_wheel_vel, rot_wheel_vel))
            lv, rv = operate.pibot.set_velocity(move, tick=lin_wheel_vel, turning_tick=rot_wheel_vel, time=dt)
        else:
            print_toggleable('Warning: Unidentified move command')
    
    if turn_drive == 0:
        time.sleep(0.3) # wait to prevent blur
    operate.take_pic()

    drive_meas = Drive(lv, rv, dt)
    lms = operate.update_slam(drive_meas)

    # running detector and saving detector
    if args.targets_detection == 1:
        operate.command['inference'] = True
    
    # do not save raw images during proper run
    if args.save_img == 1:
        operate.command['save_image'] = True 
        operate.save_image()

    if args.targets_detection == 1:
        num_targets, detected_targets = operate.detect_target()
        
        operate.fruitPoseEst.label_img = operate.detector_output
        operate.fruitPoseEst.label_robot_pose = operate.get_robot_pose()
        
        for target in detected_targets:
            if target in operate.fruitPoseEst.unconfirmed_fruits_idx_list:
                operate.reset_path = True
    else:
        num_targets = 0

    if num_targets > 0 or force_est: # only recalculate if something is detected
            operate.fruitPoseEst.refresh_est_pose_local()
        
    # visualise
    update_display()
    
    return lms

# spin 360° to localise the robot
def spin_localisation(div = 8, quick_spin = False, force_complete = False):
    robot_pose = operate.get_robot_pose()
    total_rotation = 2*math.pi + robot_pose[2]
    
    if quick_spin:
        operate.notification = 'Aruco-spin-localising...'
    else:
        operate.notification = 'Spin-localising...'

    for d in reversed(range(div+1)):
        robot_pose = operate.get_robot_pose()
        if d != 0:
            pose_to_rot = (total_rotation - robot_pose[2])/d
        else: 
            pose_to_rot = (total_rotation - robot_pose[2])
        pose_to_rot = clamp_angle(pose_to_rot)
        anticlockwise = np.sign(pose_to_rot)
        pose_to_rot = abs(pose_to_rot)
        turn_time = pose_to_rot/2*operate.baseline/(operate.scale*operate.wheel_vel)
        if quick_spin:
            print_toggleable("Aruco-spin localising: Turning for {:.4f} seconds for {:.4f} rad, {}".format(turn_time, pose_to_rot, d))
        else:
            print_toggleable("Spin localising: Turning for {:.4f} seconds for {:.4f} rad".format(turn_time, pose_to_rot))
        lms = move_and_update(wheel_vel = operate.wheel_vel, dt = turn_time, turn_drive = 0, dir = anticlockwise)

        if args.ir_backoff == 1:
            lms = ir_backoff(lms_prev=lms)
            if operate.reset_path and not force_complete:
                return lms
        
        if quick_spin: # quick spin mode: stop immediately when an aruco marker is detected
            if len(lms) > 0:
                return lms
            
    return lms

def get_nearest_aruco(one_dir = False, dir = 1, searched_markers = []):
    robot_pose = operate.get_robot_pose()
    
    # calculate angle to fruits
    fruits_pose_list, _ = operate.fruitPoseEst.get_search_order_local()
    fruits_angle_diff = 1000*np.ones(len(fruits_pose_list))
    for idx, fruit_pose in enumerate(fruits_pose_list):
        dist = math.hypot(fruit_pose[1] - robot_pose[1], fruit_pose[0] - robot_pose[0])
        if dist > 0.8: # not a problem if it is too far away
            fruits_angle_diff[idx] = 1000 # some high number
        else:
            if one_dir: # rotate anticlockwise only
                if dir > 0:
                    fruits_angle_diff[idx] = clamp_angle(math.atan2(fruit_pose[1] - robot_pose[1], fruit_pose[0] - robot_pose[0]) - robot_pose[2], 0, 2*np.pi)
                else:
                    fruits_angle_diff[idx] = clamp_angle(math.atan2(fruit_pose[1] - robot_pose[1], fruit_pose[0] - robot_pose[0]) - robot_pose[2], -2*np.pi, 0)
            else:
                fruits_angle_diff[idx] = clamp_angle(math.atan2(fruit_pose[1] - robot_pose[1], fruit_pose[0] - robot_pose[0]) - robot_pose[2])
        
    angle_diff = 1000*np.ones(10)
    for idx, aruco_pos in enumerate(aruco_true_pos):
        dist = math.hypot(aruco_pos[1] - robot_pose[1], aruco_pos[0] - robot_pose[0])
        if dist > 0.4: # do not turn to arucos which are too close
            if one_dir: # rotate anticlockwise only
                if dir > 0:
                    angle_diff[idx] = clamp_angle(math.atan2(aruco_pos[1] - robot_pose[1], aruco_pos[0] - robot_pose[0]) - robot_pose[2], 0, 2*np.pi)
                else:
                    angle_diff[idx] = clamp_angle(math.atan2(aruco_pos[1] - robot_pose[1], aruco_pos[0] - robot_pose[0]) - robot_pose[2], -2*np.pi, 0)
            else:
                angle_diff[idx] = clamp_angle(math.atan2(aruco_pos[1] - robot_pose[1], aruco_pos[0] - robot_pose[0]) - robot_pose[2])
        
        # remove if a target is potentially blocks the sight
        fruit_aruco_dist = abs(fruits_angle_diff - angle_diff[idx].item())
        if np.any(fruit_aruco_dist < 0.1745): # 10 degrees
            angle_diff[idx] = 1000 # some high number
            
        if idx in searched_markers:
            angle_diff[idx] = 1000 # do not turn to the same marker
        
    order = np.argsort(abs(angle_diff))
    angle_diff_sorted = angle_diff[order]
    return angle_diff_sorted, order

def aruco_localisation(curlms = []):
    operate.notification = 'Aruco-localising...'

    if len(curlms) != 0:
        return

    aruco_pos_list, order = get_nearest_aruco()
    searched_markers = []
    
    print('Searching for aruco')
    # turn to the closest only
    pose_to_rot = aruco_pos_list[0].item()
    pose_to_rot = clamp_angle(pose_to_rot)
    anticlockwise = np.sign(pose_to_rot)
    pose_to_rot = abs(pose_to_rot)
    turn_time = pose_to_rot/2*operate.baseline/(operate.scale*operate.wheel_vel)
    print_toggleable("Aruco-localising: Turning for {:.2f} seconds for {:.2f} rad towards marker {}".format(turn_time, anticlockwise*pose_to_rot, order[0]+1))
    lms = move_and_update(wheel_vel = operate.wheel_vel, dt = turn_time, turn_drive = 0, dir = anticlockwise)
    searched_markers.append(order[0])
    if args.ir_backoff == 1:
        lms = ir_backoff(lms_prev = lms)

    aruco_localise_count = 0
    while len(lms) == 0: 
        # lms = spin_localisation(div = 12, quick_spin = True)
        aruco_pos_list, order = get_nearest_aruco(one_dir = False, dir = anticlockwise, searched_markers=searched_markers)
        # turn to random closest
        random_order = random.randint(0, 1)
        pose_to_rot = aruco_pos_list[random_order].item()
        while abs(pose_to_rot) < 0.1742665: # choose the next one if less than 10°
            random_order += 1
            pose_to_rot = aruco_pos_list[random_order].item()
        searched_markers.append(order[random_order])
        
        if abs(pose_to_rot) > 500: # escape if the selected pose_to_rot is invalid
            pose_to_rot = aruco_pos_list[0].item()

        if abs(pose_to_rot) <= 500:
            pose_to_rot = clamp_angle(pose_to_rot)
            anticlockwise = np.sign(pose_to_rot)
            pose_to_rot = abs(pose_to_rot)
            turn_time = pose_to_rot/2*operate.baseline/(operate.scale*operate.wheel_vel)
            print_toggleable("Aruco-localising {}: Turning for {:.2f} seconds for {:.2f} rad towards marker {}".format(aruco_localise_count, turn_time, anticlockwise*pose_to_rot, order[random_order]+1))
            lms = move_and_update(wheel_vel = operate.wheel_vel, dt = turn_time, turn_drive = 0, dir = anticlockwise)
        aruco_localise_count += 1

        if aruco_localise_count > 1:
            while len(lms) == 0:
                lms = spin_localisation(div = 6, quick_spin = True)

def drive_straight(dist, dir): 
    # after turning, drive straight to the waypoint
    drive_time = dist/(operate.scale*operate.wheel_vel)  # replace with your calculation
    print_toggleable("Driving for {:.2f} seconds for {:.2f}m".format(drive_time, dist))
    lms = move_and_update(wheel_vel = operate.wheel_vel, dt = drive_time, turn_drive = 1, dir = dir)
    
    return lms

def ir_backoff(lms_prev = []):
    dr, dl = operate.pibot.ir_detect()
    if dr == 0 or dl == 0:
        lms = drive_straight(dist = 0.2, dir = -1) # drive backwards
        operate.reset_path = True # toggle reset path flag
        operate.reached = False
        return lms
        
    return lms_prev 

def us_backoff(lms_prev = []):
    dist = operate.pibot.us_distance()
    while dist < 0.4:
        lms = drive_straight(dist = 0.1, dir = -1) # drive backwards
        operate.reset_path = True # toggle reset path flag
        operate.reached = False
        return lms
        
    return lms_prev

def create_waypoints_surface(waypoints_list, waypoint_pics):
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
    fruits_surface = operate.fruitPoseEst.draw_local(res = res, m2pixel = m2pixel)
    canvas.blit(fruits_surface, map_location)
    
    pygame.display.update()

def create_obstacle_map(astar_obstacle_file_path, res, sz = (3.2, 3.2)):
    m2pixel = res[1]/sz[1]
    
    map = 255*np.ones(res).astype(np.uint8)

    # borders
    border_width = 0.05
    corner_cube = to_im_coor((-sz[0]/2+border_width, sz[1]/2-border_width), res, m2pixel)
    top_left_coor = to_im_coor((-sz[0]/2, sz[1]/2), res, m2pixel)
    top_right_coor = to_im_coor((sz[0]/2, sz[1]/2), res, m2pixel)
    bottom_left_coor = to_im_coor((-sz[0]/2, -sz[1]/2), res, m2pixel)
    bottom_right_coor = to_im_coor((sz[0]/2, -sz[1]/2), res, m2pixel)
    
    map[top_left_coor[1]:bottom_left_coor[1],top_left_coor[0]:(top_left_coor[0] + corner_cube[0])] = 0 # left border 
    map[top_right_coor[1]:bottom_right_coor[1],(top_right_coor[0] - corner_cube[0]):top_right_coor[0]] = 0 # right border 
    map[top_left_coor[1]:(top_left_coor[1] + corner_cube[1]),top_left_coor[0]:top_right_coor[0]] = 0 # top border 
    map[(bottom_right_coor[1] - corner_cube[1]):bottom_right_coor[1],bottom_left_coor[0]:bottom_right_coor[0]] = 0 # bottom border

    object_clearance = args.obstacle_dist
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
    fruits_locations, _ = operate.fruitPoseEst.get_search_order_local()
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

def find_closest_at_target(target_pose, robot_pose, clearance = None):
    mid_target = target_pose.copy()
    
    if clearance is None:
        clearance = obstacle_radius
    
    dist = math.hypot(target_pose[1] - robot_pose[1], target_pose[0] - robot_pose[0])
    if dist < clearance:
        mid_target[0] = robot_pose[0]
        mid_target[1] = robot_pose[1]
        return mid_target
    

    delta_pose = math.atan2(target_pose[1] - robot_pose[1], target_pose[0] - robot_pose[0])
    mid_target[0] = target_pose[0] - clearance*np.cos(delta_pose)
    mid_target[1] = target_pose[1] - clearance*np.sin(delta_pose)
    init_delta_pose = delta_pose
    while not check_no_obstruction(mid_target):
        delta_pose += 0.08727 # increase delta_pose by 5°
        mid_target[0] = target_pose[0] - clearance*np.cos(delta_pose)
        mid_target[1] = target_pose[1] - clearance*np.sin(delta_pose)
        
        if abs(delta_pose) >= init_delta_pose + 2*np.pi: # give up searching if a circle around it is all obstructed
            print_toggleable('Cannot find unobstructed closest point, increasing clearance')
            clearance += 0.03
    
    return mid_target

def check_no_obstruction(mid_target):
    target_x, target_y = to_im_coor((mid_target[0], mid_target[1]), res_a_star, m2pixel=res_a_star[0]/sz[0])
    
    if obstacle_map[target_y, target_x] == 0:
        return False
    
    return True

def create_path_surface(a_star_waypoints, res, res_a_star):
    path_surface = pygame.Surface([500, 500], pygame.SRCALPHA, 32) # transparent surface
    if len(a_star_waypoints) == 0:
        return path_surface

    for xy in a_star_waypoints:
        xy_projected = (int(1.0*xy.x/res_a_star[0]*res[0]), int(1.0*xy.y/res_a_star[1]*res[1]))
        path_surface.blit(pellet_img, (xy_projected[0] - pellet_img.get_width()//2, xy_projected[1] - pellet_img.get_height()//2))
    
    return path_surface

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='map.txt')
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
    parser.add_argument("--verbose", metavar='', type=int, default=1)
    parser.add_argument("--noisy", metavar='', type=int, default=0)
    parser.add_argument("--aruco_localise", metavar='', type=int, default=1)
    parser.add_argument("--obstacle_dist", metavar='', type=float, default=0.2)
    parser.add_argument("--ir_backoff", metavar='', type=int, default=1)
    parser.add_argument("--wheel_vel", metavar='', type=int, default=30)
    parser.add_argument("--targets_detection", metavar='', type=int, default=0)
    parser.add_argument("--spin", metavar='', type=int, default=0)
    parser.add_argument("--us_backoff", metavar='', type=int, default=0)
    args, _ = parser.parse_known_args()

    # GUI
    text_colour = (220, 220, 220)

    width, height = 880, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2022 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading_large_phase_2.png')
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
                canvas.blit(pygame.image.load('pics/8bit/explosion.png'), (x_-100, 565-100))
            else:
                if fire_count%10 == 0:
                    rand_x = random.randint(int(800-fire_count/10), width-10)
                    rand_y = random.randint(int(565-fire_count/10), height-10)
                    rand_int = random.randint(1, 5)
                    canvas.blit(pygame.image.load(f'pics/8bit/fire_{rand_int}.png'), (rand_x, rand_y))
            pygame.display.update()
            fire_count += 1
    
    canvas.blit(splash, (0, 0))

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    operate = Operate(args, fruits_list, fruits_true_pos, search_list, auto=True)
    operate.ekf_on = True # turn ekf on
    operate.wheel_vel = args.wheel_vel
    operate.no_delay = False
    operate.extra_verbose = False
    operate.fruitPoseEst.refresh_est_pose_local()
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
 
    operate.ekf.populate_landmarks(aruco_true_pos)
    
    obstacle_radius = math.hypot(args.obstacle_dist, args.obstacle_dist) + 0.01
    astar_obstacle_file_path = "astar_obstacles.json"
    res_a_star = (100, 100)
    obstacle_map = create_obstacle_map(astar_obstacle_file_path, res_a_star)
    a_star_waypoints = []
    
    waypoint = [0.0, 0.0]
    robot_pose = [0.0, 0.0, 0.0]
    waypoints_list = []
    waypoint_idx_increment = args.waypoint_inc
    
    # update once before drawing
    move_and_update(wheel_vel=0, dt = 0, turn_drive = 0, force_est = True)
    
    # drawing annotations
    update_display()

    waypoint_idx = 0
    target_idx = 0
    targets_list = []
    targets_list, fruits = operate.fruitPoseEst.get_search_order_local() # initialise targets_list
    
    # calculate origin location in A* map resolution
    origin_x, origin_y = to_im_coor((0.0, 0.0), res_a_star, m2pixel=res_a_star[1]/sz[1])

    # Mandatory spin at the start
    if args.spin == 1:
        spin_localisation(div=6, force_complete=True)
        spin_count = 0
        while operate.reset_path: # ensure an unobstructed, complete spin
            operate.reset_path = False
            spin_localisation(div=6, force_complete = True)
            spin_count += 1
            print_toggleable(f'Spinning count: {spin_count} ')
            if spin_count > 2:
                operate.reset_path = False
                print_toggleable('Spinning aborted. Max spin count reached. ')
    mandatory_spin_done = True
    stop_flag = False
    operate.reached = False
    while True:
        x,y = 0.0,0.0
        
        # enter the waypoints
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
                        spin_localisation(div=6)
                    
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
            
            operate.reached = False
            # checking if the target is reached before navigating
            if len(targets_list) !=0:
                try:
                    target_pose = targets_list[target_idx]
                except:
                    stop_flag = True
                    target_idx = target_idx - 1
                    target_pose = targets_list[target_idx]
                    print_toggleable('No more detected targets') 

                dist = math.hypot(target_pose[1] - robot_pose[1], target_pose[0] - robot_pose[0])
                if dist < 0.25:
                    operate.reached = True
            
            if nav_type == 1:
                # Only plan before moving to a new target, but overridable
                if operate.reset_path: # reset paths if new targets are found
                    waypoint_idx = 0
                    waypoints_list = [] # clear waypoints_list
                    print_toggleable('Unknown targets found / Obstacle detected. Resetting path. ')

                if len(waypoints_list) == 0: # first waypoint 
                    if not operate.reset_path: 
                        if mandatory_spin_done: # skip spinning if mandatory spin is done
                            mandatory_spin_done = False
                        else:
                            if args.spin == 1:
                                spin_localisation(div=6, force_complete = True) # spin and look for additional fruits
                                spin_count = 0
                                while operate.reset_path: # ensure an unobstructed, complete spin
                                    operate.reset_path = False
                                    spin_localisation(div=6, force_complete = True)
                                    print_toggleable('Unknown targets found / Obstacle detected during spin. Resetting path. ')
                                    spin_count += 1
                                    if spin_count > 2:
                                        operate.reset_path = False
                                        print_toggleable('Spinning aborted. Max spin count reached. ')
                    else:
                        operate.reset_path = False # do not spin again if new fruits are found
                    
                    targets_list, targets = operate.fruitPoseEst.get_search_order_local()
                    try:
                        target_pose = targets_list[target_idx]
                    except:
                        stop_flag = True
                        target_idx = target_idx - 1
                        target_pose = targets_list[target_idx]
                        print_toggleable('No more detected targets') 
                    # path planning
                    close_target = find_closest_at_target(target_pose, robot_pose)
                    robot_x, robot_y = to_im_coor((robot_pose[0], robot_pose[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                    close_target_x, close_target_y = to_im_coor((close_target[0], close_target[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                    
                    a_star_waypoints = []
                    first_path = True
                    while len(a_star_waypoints) == 0:
                        if not first_path: # preventing unecessary spins
                            print_toggleable('Spinning because path cannot be found')
                            if args.spin == 1:
                                spin_localisation(div = 6) # keep localising until a path is found
                            else:
                                if args.aruco_localise == 1:
                                    aruco_localisation()
    
                        operate.notification = 'Finding path...'
                        update_display()
                        robot_pose = operate.get_robot_pose()
                        
                        # force move the robot out of obstacle area
                        if not check_no_obstruction(robot_pose):
                            escape_robot_pose = find_closest_at_target(robot_pose, close_target)
                            robot_pose[0] = escape_robot_pose[0]
                            robot_pose[1] = escape_robot_pose[1]
                            print_toggleable('Robot perceived within obstacle, force-teleporting robot')
                        
                        robot_x, robot_y = to_im_coor((robot_pose[0], robot_pose[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                        obstacle_map = create_obstacle_map(astar_obstacle_file_path, res_a_star)
                        a_star = initialiseAStar(astar_obstacle_file_path, res_a_star, start = [robot_x, robot_y], end = [close_target_x, close_target_y])
                        a_star_waypoints = a_star.main()
                        a_star_waypoints.reverse() # the list provided by astar is reversed   
                        first_path = False
                        
                        if len(a_star_waypoints) != 0:
                            waypoint_idx -= waypoint_idx_increment # ensure the first waypoint is at start location
                                           

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
                    operate.reached = True
                
                operate.notification = f'Moving to target {target_idx} waypoint {waypoint_idx} at {x}, {y}'
            
            elif nav_type == 2:
                # continuous path planning
                try:
                    targets_list, targets = operate.fruitPoseEst.get_search_order_local()
                    target_pose = targets_list[target_idx]
                except:
                    stop_flag = True
                    target_idx = target_idx - 1
                    print_toggleable('No more detected targets') 
                
                close_target = find_closest_at_target(target_pose, robot_pose)
                robot_x, robot_y = to_im_coor((robot_pose[0], robot_pose[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                close_target_x, close_target_y = to_im_coor((close_target[0], close_target[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                
                operate.notification = 'Finding path...'
                update_display()
                
                obstacle_map = create_obstacle_map(astar_obstacle_file_path, res_a_star)
                a_star = initialiseAStar(astar_obstacle_file_path, res_a_star, start = [robot_x, robot_y], end = [close_target_x, close_target_y])
                a_star_waypoints = a_star.main()
                a_star_waypoints.reverse() # the list provided by astar is reversed

                update_display() # display path
                
                skip_idx = waypoint_idx_increment # project forward since each step only move the robot by 1 pixel, or 3.2mm x 3 is about 0.1m
                if skip_idx > len(a_star_waypoints)-1:
                    operate.reached = True
                    skip_idx = len(a_star_waypoints) - 1
                    
                if len(a_star_waypoints) == 0: # stay at current position
                    operate.reached = True
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
            spin_localisation(div=6)
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
                        spin_localisation(div=6)
                
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
            #             drive_straight(0.4, 1)
                
        update_display()

        # estimate the robot's pose
        robot_pose = operate.get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint, robot_pose)
        robot_pose = operate.get_robot_pose()
        print_toggleable("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        move_and_update(wheel_vel=0, dt = 0, turn_drive = 0) # stop the robot
    
        if operate.reached: # if target is reached
            target_idx += 1 # next target
            waypoint_idx = 0
            waypoints_list = [] # clear waypoints_list
            print_toggleable(f'Reached target {target_idx}, pausing for 3 seconds')
            for i in reversed(range(3)): # stay in place for 3 seconds
                operate.notification = f'Reached target {target_idx}, resuming in {i+1}'
                if args.noisy == 1:
                    operate.pibot.honk()
                pygame.event.get() # preventing the window from crashing
                update_display()
                pygame.time.wait(1000) 

        if args.level != 1 and (target_idx >= len(search_list) or stop_flag):
            while True:
                operate.notification = f'Done navigating. (Visited all {len(search_list)} targets)' #Done navigating. (Visited all 3 targets)'
                if args.noisy == 1:
                    operate.pibot.play_music()
                    operate.pibot.led_on()
                print_toggleable(f'Done navigating. (Visited all {len(search_list)} targets), Time remaining: {operate.count_down - time.time() + operate.start_time}')

                if stop_flag:
                    operate.notification = 'Done navigating. (No more target detected)'
                    print_toggleable('Done navigating. (No more target detected)')
                update_display()
                
                events = pygame.event.get() # prevent window from crashing
                for event in events:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                
        if args.return_origin == 1:
            if operate.reached:
                # go back to origin
                robot_x, robot_y = to_im_coor((robot_pose[0], robot_pose[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                
                a_star_waypoints = []
                waypoint_idx = 0
                
                while len(a_star_waypoints) == 0:
                    aruco_localisation()
                    # return to origin
                    operate.notification = 'Finding path back to origin...'
                    update_display()
                    robot_pose = operate.get_robot_pose()
                    robot_x, robot_y = to_im_coor((robot_pose[0], robot_pose[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                    obstacle_map = create_obstacle_map(astar_obstacle_file_path, res_a_star)
                    a_star = initialiseAStar(astar_obstacle_file_path, res, start = [robot_x, robot_y], end = [origin_x, origin_y])
                    a_star_waypoints = a_star.main()
                    a_star_waypoints.reverse()
                    
                    if len(a_star_waypoints) == 0:
                        print_toggleable('Fail to find path')
                    else:
                        waypoint_idx -= waypoint_idx_increment # ensure the first waypoint is at start location
                
                operate.origin_reached = False
                while not operate.origin_reached:
                    operate.notification = 'Navigating back to origin'
                    update_display()
                    
                    if operate.reset_path: # reset paths if new targets are found
                        waypoint_idx = 0
                        waypoints_list = [] # clear waypoints_list
                        print_toggleable('Unknown targets found. Re-running path finding. ')
                    
                    first_path = True
                    while len(a_star_waypoints) == 0:
                        if not first_path:
                            if not operate.reset_path:
                                aruco_localisation()
                            else:
                                operate.reset_path = False
                            
                        # return to origin
                        operate.notification = 'Finding path back to origin...'
                        update_display()
                        robot_pose = operate.get_robot_pose()
                        robot_x, robot_y = to_im_coor((robot_pose[0], robot_pose[1]), res_a_star, m2pixel=res_a_star[1]/sz[1])
                        
                        obstacle_map = create_obstacle_map(astar_obstacle_file_path, res_a_star)
                        a_star = initialiseAStar(astar_obstacle_file_path, res, start = [robot_x, robot_y], end = [origin_x, origin_y])
                        a_star_waypoints = a_star.main()
                        a_star_waypoints.reverse()
                        
                        first_path = False
                        if len(a_star_waypoints) == 0:
                            print_toggleable('Fail to find path')
                        else:
                            waypoint_idx -= waypoint_idx_increment # ensure the first waypoint is at start location
                    
                    waypoint_idx += waypoint_idx_increment
                    if waypoint_idx >= len(a_star_waypoints)-1:
                        waypoint_idx = len(a_star_waypoints)-1
                        operate.origin_reached = True
                    
                    xy = to_m((a_star_waypoints[waypoint_idx].x, a_star_waypoints[waypoint_idx].y), res_a_star, offset = (0, 0), m2pixel=res_a_star[1]/sz[1])
                    x = xy[0]
                    y = xy[1]
                    
                    waypoint = [x, y]
                    waypoints_list.append([x, y])
                    
                    drive_to_point(waypoint, robot_pose)
                    robot_pose = operate.get_robot_pose()
                    
                    move_and_update(wheel_vel=0, dt = 0, turn_drive = 0) # stop the robot
                    
                # reset waypoints
                waypoint_idx = 0
                waypoints_list = [] # clear waypoints_list