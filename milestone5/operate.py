# teleoperate the robot, perform SLAM and object detection

# basic python packages
import numpy as np
import cv2 
import os, sys
import time
import random 
import json
import math
import traceback
import requests

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
from util.measure import Drive
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
from util.helper import clamp, to_m, to_im_coor, clamp_angle

import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
# resnet5
# sys.path.insert(0,"{}/network/".format(os.getcwd()))
# sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
# from network.scripts.detector import Detector # resnet

# yolov5
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector_yolov5 import Detector # yolov5

# yolov7
# sys.path.insert(0,"{}/network7/".format(os.getcwd()))
# sys.path.insert(0,"{}/network7/scripts".format(os.getcwd()))
# from network7.scripts.detector_yolov7 import Detector # yolov7

from TargetPoseEst import FruitPoseEst

pygame.font.init() 
TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
LABEL_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 15)
ALT_TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

class Operate:
    def __init__(self, args, confirmed_fruits_list = [], confirmed_fruits_locations = [], search_list = [], auto=False):
        self.folder = 'pibot_dataset/'
        # if not os.path.exists(self.folder):
        #     os.makedirs(self.folder)
        # else:
        #     shutil.rmtree(self.folder)
        try:
            os.makedirs(self.folder)
        except:
            pass
        
        # if not os.path.exists('lab_output/'):
        #     os.makedirs('lab_output/')
        # else:
        #     shutil.rmtree('lab_output/')
        #     
        try:
            os.makedirs('lab_output/')
        except:
            pass
        
        self.backup = False
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = Alphabot(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip, auto=auto)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None

        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = True # automatic ekf on
        self.lock_marker_double_reset_confirm = 0
        self.unlock_marker_double_reset_confirm = 0
        self.double_reset_confirm = 0
        self.image_id = 0

        self.notification = 'Starting... (Or the window has crashed)'
        self.default_notification = 'Teleoperating'
        self.prev_notification = ''
        self.extra_verbose = True
        
        # a 10min timer
        self.count_down = 600
        self.start_time = time.time()
        self.control_clock = time.time()
        self.timer_pic = pygame.image.load('pics/8bit/timer.png')
        
        # initialise images
        self.camera_off_img = cv2.imread('pics/8bit/camera_off_splash.png')
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.ckpt, use_gpu=False)
            self.network_vis = np.ones((240, 320,3))* 100
        self.bg = pygame.image.load('pics/gui_mask_large.png')
        
        # Flags
        self.reset_path = False
        self.reached = False
        self.origin_reached = False
        self.moving = False

        self.speed_img = pygame.image.load('pics/8bit/speed.png')
        self.wheel_vel = 30 # tick to move the robot, higher means faster
        self.MIN_WHEEL_VEL = 10
        self.MAX_WHEEL_VEL = 80
        
        # imports camera / wheel calibration parameters
        fileS = "calibration/param/scale.txt"
        self.scale = np.loadtxt(fileS, delimiter=',')
        fileB = "calibration/param/baseline.txt"
        self.baseline = np.loadtxt(fileB, delimiter=',')
        
        self.small_dist = 0.4 # moving a short distance multiple time to reach a longer distance
        self.dist_tolerance = 0.05 # tolerance to define whether the point is reached
        self.no_delay = True
        self.camera_on = True
        self.spin_div_img = pygame.image.load('pics/8bit/spin_div.png')
        self.spin_div = 12
        self.spin_div_list = [2, 3, 4, 6, 8, 10, 12, 16, 18, 36, 72, 90, 180, 360]
        self.spin_div_idx = self.spin_div_list.index(self.spin_div)
        self.MAX_SPIN_DIV_IDX = len(self.spin_div_list) - 1
        
        self.trans_dist_img = pygame.image.load('pics/8bit/trans_dist.png')
        self.trans_dist = 0.1
        self.trans_dist_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.trans_dist_idx = self.trans_dist_list.index(self.trans_dist)
        self.MAX_TRANS_DIST_IDX = len(self.trans_dist_list) - 1
        
        self.steer_img = pygame.image.load('pics/8bit/steer.png')
        self.steer = 0.1
        self.steer_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        self.steer_idx = self.steer_list.index(self.steer)
        self.MAX_STEER_IDX = len(self.steer_list) - 1
        
        self.music_counter = 1
        
        self.led_on = False
        self.pibot.led_off()

        self.fruitPoseEst = FruitPoseEst(confirmed_fruits_list = confirmed_fruits_list, confirmed_fruits_locations = confirmed_fruits_locations, search_list = search_list)
        self.estimator_enabled = False
        self.targets_double_reset_confirm = 0
        
        self.waypoints = []
        self.waypoint_pics = []
        for i in range(1, 11):
            f_ = f'./pics/8bit/waypoint ({i}).png'
            self.waypoint_pics.append(pygame.image.load(f_))

        self.web_on = False
        self.cross_img = pygame.image.load('./pics/8bit/cross.png')

    def set_notification(self, string, update_buffer = False):
        if self.extra_verbose:
            if update_buffer:
                self.prev_notification = self.notification
            self.notification = string
    
    # wheel control
    def control(self):    
        if self.command['motion'][0] == 0 and self.command['motion'][1] == 0:
            # only send the request to stop once
            if self.moving:
                lv, rv = self.pibot.set_velocity(
                self.command['motion'], tick = self.wheel_vel, turning_tick=self.wheel_vel)
                self.moving = False
            else:
                lv, rv = 0, 0
        else:
            self.moving = True
            if args.play_data:
                lv, rv = self.pibot.set_velocity()      
            else:
                lv, rv = self.pibot.set_velocity(
                    self.command['motion'], tick = self.wheel_vel, turning_tick=self.wheel_vel)
            if not self.data is None:
                self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        
        if not self.pibot.success and self.backup:
            print('Pibot command failed, backup-ing')
            self.command['output'] = True
            self.record_data()
            self.fruitPoseEst.refresh_est_pose_local()
            self.fruitPoseEst.save_targets()
            self.save_compiled_map()
            self.save_compiled_map()
            
        return drive_meas

    # camera control
    def take_pic(self):
        if self.camera_on:
            self.img = self.pibot.get_image()
            if not self.pibot.success and self.backup:
                print('Pibot command failed, backup-ing')
                self.command['output'] = True
                self.record_data()
                self.fruitPoseEst.refresh_est_pose_local()
                self.fruitPoseEst.save_targets()
                self.save_compiled_map()

            if not self.data is None:
                self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)

        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.set_notification('Robot pose is successfuly recovered')
                self.ekf_on = True
            else:
                self.set_notification('Recover failed, need >2 landmarks!')
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)
        
        return lms

    def get_robot_pose(self):
        all_states = self.ekf.get_state_vector().reshape(1, -1)
        robot_pose = list([all_states[0][0], all_states[0][1], all_states[0][2]])
        
        return robot_pose
    
    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img, self.aruco_img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            
            return len(np.unique(self.detector_output))-1, np.unique(self.detector_output).tolist()
        return 0, []

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            if not self.pibot.success:
                print('Pibot command failed, backup-ing')
                self.command['output'] = True
                self.record_data()
                self.fruitPoseEst.refresh_est_pose_local()
                self.fruitPoseEst.save_targets()
                self.save_compiled_map()
                self.save_compiled_map()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.set_notification(f'{f_} is saved')

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip, auto=False):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot, auto=auto)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.set_notification('SLAM Map is saved')
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                
                self.set_notification(f'Prediction is saved to {self.pred_fname}')
            else:
                self.set_notification(f'No prediction in buffer, save ignored')
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state_world(res=(500, 460+v_pad),
            not_pause = self.ekf_on)
        
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
    
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notification = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notification, (h_pad+10, 596))
        
        time_remain = self.count_down - time.time() + self.start_time
        if time_remain < 300 and time_remain > 290:
            self.default_notification = 'Start realigning and searching targets now'
            self.set_notification(self.default_notification)
        if time_remain > 0:
            time_remain = f'{time_remain:03.0f}s'
            text_color = (50, 50, 50)
        elif int(time_remain)%2 == 0:
            time_remain = "TIME!!!"
            self.default_notification = '10 minutes exceeded -- Proceed now'
            self.set_notification(self.default_notification)
            text_color = (255, 0, 0)
        else:
            time_remain = ""
            self.default_notification = ''
            self.set_notification(self.default_notification)
            text_color = (255, 0, 0)
        
        count_down_surface = TEXT_FONT.render(time_remain, False, text_color, (255, 255, 255))
        canvas.blit(self.timer_pic, (2*h_pad+320+5, 540))
        canvas.blit(count_down_surface, (2*h_pad+320+33, 540))
        
        # draw speed indicator
        canvas.blit(self.speed_img, (880-h_pad-self.speed_img.get_width()-30-15, 540))
        speed_text = ALT_TEXT_FONT.render(f'{self.wheel_vel}', False, (50, 50, 50), (220, 220, 220))
        canvas.blit(speed_text, (880-h_pad-self.speed_img.get_width()-15, 540))
        
        # draw spin div indicator
        canvas.blit(self.spin_div_img, (880-h_pad-self.spin_div_img.get_width()-30-120, 540))
        spin_div_text = ALT_TEXT_FONT.render(f'{360//self.spin_div:.0f}°', False, (50, 50, 50), (220, 220, 220))
        canvas.blit(spin_div_text, (880-h_pad-self.spin_div_img.get_width()-30-90, 540))
        
        # draw trans_dist indicator
        canvas.blit(self.trans_dist_img, (880-h_pad-self.trans_dist_img.get_width()-30-205, 540))
        text = f'{self.trans_dist:.2f}'[1:]
        trans_dist_text = ALT_TEXT_FONT.render(text, False, (50, 50, 50), (220, 220, 220))
        canvas.blit(trans_dist_text, (880-h_pad-self.trans_dist_img.get_width()-30-175, 540))
        
        # draw steer indicator
        canvas.blit(self.steer_img, (880-h_pad-self.steer_img.get_width()-30-295, 540))
        text = f'{self.steer:.2f}'[1:]
        steer_text = ALT_TEXT_FONT.render(text, False, (50, 50, 50), (220, 220, 220))
        canvas.blit(steer_text, (880-h_pad-self.steer_img.get_width()-30-265, 540))
    
        # draw web
        if self.web_on:
            web_surface = self.create_web_surface()
            canvas.blit(web_surface, (2*h_pad+320, v_pad))
            
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    def realign_map(self):
        # cur_robot_pose = self.get_robot_pose()
        #self.fruitPoseEst.realign_map(cur_robot_pose)
        self.ekf.realign_map()
    
    def save_compiled_map(self):    
        phase1_map_dict = {}
        for idx, tag in enumerate(self.ekf.taglist):
            pose = {
                "x": self.ekf.markers[0][idx],
                "y": self.ekf.markers[1][idx]
            }
            phase1_map_dict[f"aruco{tag}_0"] = pose
        
        fruits_locations, fruits = self.fruitPoseEst.get_search_order_local()
        for idx, locations in enumerate(fruits_locations):
            fruit = fruits[idx]
            pose = {
                "x": locations[0],
                "y": locations[1]
            }
            phase1_map_dict[f"{fruit}_0"] = pose
    
        with open('map.txt', 'w') as phase1_map_f:
            json.dump(phase1_map_dict, phase1_map_f, indent=2)

    # keyboard controls      
    def update_keyboard(self):
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LSHIFT]:
                self.update_mouse()
            else:
                # drive forward
                if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    self.command['motion'] = [1,0] # TODO: replace with your code to make the robot drive forward
                    # steering
                    if event.mod & pygame.KMOD_LCTRL:
                        self.command['motion'] = [1, self.steer]
                    elif event.mod & pygame.KMOD_LALT:
                        self.command['motion'] = [1, -self.steer]
                elif event.type == pygame.KEYUP and event.key == pygame.K_UP:
                    self.command['motion'] = [0, 0]
                # drive backward
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    self.command['motion'] = [-1, 0] # TODO: replace with your code to make the robot drive backward
                    # steering
                    if event.mod & pygame.KMOD_LCTRL:
                        self.command['motion'] = [-1, -self.steer]
                    elif event.mod & pygame.KMOD_LALT:
                        self.command['motion'] = [-1, self.steer]
                elif event.type == pygame.KEYUP and event.key == pygame.K_DOWN:
                    self.command['motion'] = [0, 0]
                # steer left
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LCTRL:
                    if self.command['motion'][0] == 1 and self.command['motion'][1] == 0:
                        self.command['motion'] = [1, self.steer]
                    elif self.command['motion'][0] == -1 and self.command['motion'][1] == 0:
                        self.command['motion'] = [-1, -self.steer]
                # steer right
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LALT:
                    if self.command['motion'][0] == 1 and self.command['motion'][1] == 0:
                        self.command['motion'] = [1, -self.steer]
                    elif self.command['motion'][0] == -1 and self.command['motion'][1] == 0:
                        self.command['motion'] = [-1, self.steer]
                # steer cancel
                elif event.type == pygame.KEYUP and (event.key == pygame.K_LALT or event.key == pygame.K_LCTRL):
                    if self.command['motion'][0] == 1 and abs(self.command['motion'][1]) == self.steer:
                        self.command['motion'] = [1, 0]
                    elif self.command['motion'][0] == -1 and abs(self.command['motion'][1]) == self.steer:
                        self.command['motion'] = [-1, 0]
                # turn left
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    self.command['motion'] = [0, 1] # TODO: replace with your code to make the robot turn left
                elif event.type == pygame.KEYUP and event.key == pygame.K_LEFT:
                    self.command['motion'] = [0, 0]
                # drive right
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    self.command['motion'] = [0, -1] # TODO: replace with your code to make the robot turn right
                elif event.type == pygame.KEYUP and event.key == pygame.K_RIGHT:
                    self.command['motion'] = [0, 0]
                # stop
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.command['motion'] = [0, 0]
                # precise forward    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                    self.forward(dir = 1, dist = self.trans_dist)
                # precise backward 
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self.forward(dir = -1, dist = self.trans_dist)
                # precise rotate left
                if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                    self.rotate(dir = 1, pose_to_rot = 2*math.pi/self.spin_div)
                # precise rotate right
                if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                    self.rotate(dir = -1, pose_to_rot = 2*math.pi/self.spin_div)
                # precise 180° turn
                if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                    self.rotate(dir = 1, pose_to_rot = math.pi)
                # save image
                if event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                    self.command['save_image'] = True
                # save SLAM map
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_y:
                    self.command['output'] = True
                # reset SLAM map
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    if self.double_reset_confirm == 0:
                        self.set_notification('Press again to confirm CLEAR MAP')
                        self.double_reset_confirm +=1
                    elif self.double_reset_confirm == 1:
                        self.set_notification('SLAM Map is cleared')
                        self.double_reset_confirm = 0
                        self.ekf.reset()
                # run SLAM
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    n_observed_markers = len(self.ekf.taglist)
                    if n_observed_markers == 0:
                        if not self.ekf_on:
                            self.set_notification('SLAM is running')
                            self.ekf_on = True
                        else:
                            self.set_notification('> 2 landmarks is required for pausing')
                    elif n_observed_markers < 3:
                        self.set_notification('> 2 landmarks is required for pausing')
                    else:
                        if not self.ekf_on:
                            self.request_recover_robot = True
                        self.ekf_on = not self.ekf_on
                        if self.ekf_on:
                            self.set_notification('SLAM is running')
                        else:
                            self.set_notification('SLAM is paused')
                # run object detector
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    self.command['inference'] = True
                # save object detection outputs
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    self.command['save_inference'] = True
                # refresh est pose
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    self.fruitPoseEst.refresh_est_pose_local()
                    self.fruitPoseEst.save_targets()
                    self.set_notification('Targets map updated')
                # delete all targets location
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                    if self.targets_double_reset_confirm == 0:
                        self.set_notification('Press again to confirm CLEAR MAP')
                        self.targets_double_reset_confirm +=1
                    elif self.targets_double_reset_confirm == 1:
                        self.targets_double_reset_confirm = 0
                        self.fruitPoseEst.erase_all()
                        self.fruitPoseEst.refresh_est_pose_local()
                        self.fruitPoseEst.save_targets()
                        self.set_notification('Targets map deleted')
                # lock/unlock markers position 
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                    if event.mod & pygame.KMOD_RCTRL:
                        if self.unlock_marker_double_reset_confirm == 0:
                            self.set_notification('Press again to confirm markers unlock')
                            self.unlock_marker_double_reset_confirm +=1
                        elif self.unlock_marker_double_reset_confirm == 1:
                            self.unlock_marker_double_reset_confirm = 0
                            self.ekf.unlock_markers()
                            self.set_notification('Markers unlocked')
                    else:
                        if self.lock_marker_double_reset_confirm == 0:
                            self.set_notification('Press again to confirm markers lock')
                            self.lock_marker_double_reset_confirm +=1
                        elif self.lock_marker_double_reset_confirm == 1:
                            self.lock_marker_double_reset_confirm = 0
                            self.ekf.lock_markers()
                            self.set_notification('Markers locked')
                # lock/unlock one marker
                elif event.type == pygame.KEYDOWN and (event.key >= pygame.K_F1 and event.key <= pygame.K_F10):
                    tag = event.key - pygame.K_F1 + 1
                    
                    if event.mod & pygame.KMOD_RCTRL:
                        msg = self.ekf.unlock_one_marker(tag)
                        self.set_notification(msg)
                    elif event.mod & pygame.KMOD_RSHIFT: 
                        msg = self.ekf.delete_one_marker(tag)   
                    else:
                        msg = self.ekf.lock_one_marker(tag)
                    
                    self.set_notification(msg)  
                # lock/unlock one target
                elif event.type == pygame.KEYDOWN and (event.key >= pygame.K_1 and event.key <= pygame.K_5):
                    if event.key == pygame.K_1:
                        fruit = 'redapple'
                    elif event.key == pygame.K_2:
                        fruit = 'greenapple'
                    elif event.key == pygame.K_3:
                        fruit = 'orange'
                    elif event.key == pygame.K_4:
                        fruit = 'mango'
                    elif event.key == pygame.K_5:
                        fruit = 'capsicum'
                    
                    if event.mod & pygame.KMOD_RCTRL:
                        msg = self.fruitPoseEst.unlock_fruit(fruit)
                    elif event.mod & pygame.KMOD_RSHIFT: 
                        msg = self.fruitPoseEst.delete_fruit(fruit)
                    else:
                        msg = self.fruitPoseEst.lock_fruit(fruit)

                    self.set_notification(msg)                   
                # enable est pose
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                    self.fruitPoseEst.refresh_est_pose_local()
                    self.fruitPoseEst.save_targets()
                    self.estimator_enabled = True
                    self.set_notification('Automated target pose est enabled')
                # toggle camera on/off
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                    if self.camera_on:
                        self.camera_on = False
                        self.img = self.camera_off_img
                        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
                        self.detector_output = np.zeros([240,320], dtype=np.uint8)
                        self.set_notification('Camera off')
                    else:
                        self.camera_on = True
                        self.set_notification('Camera on')
                # save est pose, compiled map 
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    self.command['output'] = True
                    self.record_data()
                    self.fruitPoseEst.refresh_est_pose_local()
                    self.fruitPoseEst.save_targets()
                    self.save_compiled_map()
                    self.set_notification('Compiled map updated')
                # realign map
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                    self.realign_map()
                    self.command['output'] = True
                    self.record_data()
                    self.fruitPoseEst.refresh_est_pose_local()
                    self.fruitPoseEst.save_targets()
                    self.save_compiled_map()
                    self.set_notification('Map realigned and compiled map saved')
                # increase wheel_vel
                elif event.type == pygame.KEYDOWN and (event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS):
                    self.wheel_vel += 10
                    self.wheel_vel = clamp(self.wheel_vel, self.MAX_WHEEL_VEL, self.MIN_WHEEL_VEL)  
                    self.set_notification(f'Speed {self.wheel_vel}')    
                # decrease wheel_vel
                elif event.type == pygame.KEYDOWN and (event.key == pygame.K_KP_MINUS or event.key == pygame.K_MINUS):
                    self.wheel_vel -= 10
                    self.wheel_vel = clamp(self.wheel_vel, self.MAX_WHEEL_VEL, self.MIN_WHEEL_VEL)  
                    self.set_notification(f'Speed {self.wheel_vel}')    
                # decrease rotation degrees and increase spin_div
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFTBRACKET:
                    self.spin_div_idx += 1
                    self.spin_div_idx = int(clamp(self.spin_div_idx, self.MAX_SPIN_DIV_IDX, 0)) 
                    self.spin_div = self.spin_div_list[self.spin_div_idx] 
                    self.set_notification(f'Spin theta {360//self.spin_div:.0f}')    
                # increase rotation degrees and decrease spin_div
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHTBRACKET:
                    self.spin_div_idx -= 1
                    self.spin_div_idx = int(clamp(self.spin_div_idx, self.MAX_SPIN_DIV_IDX, 0)) 
                    self.spin_div = self.spin_div_list[self.spin_div_idx] 
                    self.set_notification(f'Spin theta {360//self.spin_div:.0f}')   
                # decrease translation distance
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SEMICOLON:
                    self.trans_dist_idx -= 1
                    self.trans_dist_idx = int(clamp(self.trans_dist_idx, self.MAX_TRANS_DIST_IDX, 0)) 
                    self.trans_dist = self.trans_dist_list[self.trans_dist_idx] 
                    self.set_notification(f'Trans dist {self.trans_dist}')    
                # increase translation distance
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_QUOTE:
                    self.trans_dist_idx += 1
                    self.trans_dist_idx = int(clamp(self.trans_dist_idx, self.MAX_TRANS_DIST_IDX, 0)) 
                    self.trans_dist = self.trans_dist_list[self.trans_dist_idx] 
                    self.set_notification(f'Trans dist {self.trans_dist}')    
                # decrease steer
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_PERIOD:
                    self.steer_idx -= 1
                    self.steer_idx = int(clamp(self.steer_idx, self.MAX_STEER_IDX, 0)) 
                    self.steer = self.steer_list[self.steer_idx] 
                    self.set_notification(f'Steer {self.steer}')    
                # increase steer
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SLASH:
                    self.steer_idx += 1
                    self.steer_idx = int(clamp(self.steer_idx, self.MAX_STEER_IDX, 0)) 
                    self.steer = self.steer_list[self.steer_idx] 
                    self.set_notification(f'Steer {self.steer}')        
                # play music
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    if self.music_counter%3 ==0:
                        self.pibot.play_music()
                    self.music_counter+=1
                    if self.music_counter%3 ==2:
                        self.set_notification(f'Music counter {3-self.music_counter%3}, music incoming!')
                    else:
                        self.set_notification(f'Music counter {3-self.music_counter%3}')
                # honk
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                    if event.mod & pygame.KMOD_CTRL:
                        self.pibot.meow()
                    else:
                        self.pibot.honk()
                # on/off led
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                    if self.led_on:
                        self.pibot.led_off()
                        self.led_on = False
                    else:
                        self.pibot.led_on()
                        self.led_on = True
                    if event.mod & pygame.KMOD_RCTRL:
                        self.pibot.led_red()
                        self.led_on = True
                    elif event.mod & pygame.KMOD_RSHIFT:
                        self.pibot.led_green()
                        self.led_on = True
                # IR detect
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                    dr, dl = self.pibot.ir_detect()
                    print(f'IR Detection: DR: {dr}, DL: {dl}')
                # Ultrasonic distancing
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_u:
                    dist = self.pibot.us_distance()
                    print(f'Ultrasonic Sensor: Distance: {dist}')
                # Draw web
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKQUOTE:
                    self.web_on = not self.web_on
                    if self.web_on:
                        self.set_notification('Web on')
                    else:
                        self.set_notification('Web off')
                # quit
                elif event.type == pygame.QUIT:
                    self.quit = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.quit = True
        
        if self.quit:
            pygame.quit()
            sys.exit()
            
    # mouse controls
    def update_mouse(self):
        keys = pygame.key.get_pressed()
        self.set_notification('Mouse mode: Select a waypoint', update_buffer=True)
        while keys[pygame.K_LSHIFT]:
            self.set_notification('Mouse mode: Select a waypoint')
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.spin_localisation(div=self.spin_div)
                
                if map_rect.collidepoint(pygame.mouse.get_pos()):
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
                    xy = to_m(pygame.mouse.get_pos(), res = res, offset=map_location, m2pixel=m2pixel)
                    mouse_xy_text = f'{xy[0]:3.3f}, {xy[1]:3.3f}'
                    mouse_xy_surface = TEXT_FONT.render(mouse_xy_text, False, (50, 50, 50), (255, 255, 255))
                    mouse_xy_bg = pygame.Surface((220+h_pad, mouse_xy_surface.get_height()))
                    canvas.blit(mouse_xy_bg, (width-h_pad-220, v_pad/2))
                    canvas.blit(mouse_xy_surface, (width-h_pad-220, v_pad/2))
                    update_display()
                    
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x = xy[0]
                        y = xy[1]
                        waypoint = [x, y]
                        self.waypoints.append(waypoint)
                        
                        self.set_notification(f'Moving to waypoint at {x}, {y}')
                        update_display()
                        self.move_to_waypoint(waypoint)
                        
                        self.waypoints.pop()
                        update_display()
                else:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        self.set_notification(self.default_notification)
    
    def generate_mid_waypoint(self, waypoint, robot_pose):
        dist = math.hypot(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
        mid_waypoint = waypoint.copy()
        
        if dist < self.small_dist:
            return mid_waypoint
        
        delta_pose = math.atan2(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
        mid_waypoint[0] = robot_pose[0] + self.small_dist*np.cos(delta_pose)
        mid_waypoint[1] = robot_pose[1] + self.small_dist*np.sin(delta_pose)
    
        return mid_waypoint
    
    def move_to_waypoint(self, waypoint):
        # Type 1.2: Moving after turning
        # segmented rotation
        robot_pose = self.get_robot_pose()
        dist = math.hypot(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
        
        while dist > self.dist_tolerance:
            mid_waypoint = self.generate_mid_waypoint(waypoint, robot_pose)
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
                
                anticlockwise = np.sign(pose_to_rot)
                pose_to_rot = abs(pose_to_rot)
                
                # replace with your calculation
                turn_time = pose_to_rot/2*self.baseline/(self.scale*self.wheel_vel)
                print("Turning for {:.2f} seconds for {:.2f}rad".format(turn_time, anticlockwise*pose_to_rot))
                self.move_and_update(dt = turn_time, turn_drive = 0, dir = anticlockwise)

                robot_pose = self.get_robot_pose() # update robot pose
            
            # after turning, drive straight to the waypoint
            mid_waypoint = self.generate_mid_waypoint(waypoint, robot_pose)
            mid_dist = math.hypot(mid_waypoint[1] - robot_pose[1], mid_waypoint[0] - robot_pose[0])
            drive_time = mid_dist/(self.scale*self.wheel_vel)  # replace with your calculation
            print("Driving for {:.2f} seconds for {:.2f}m".format(drive_time, mid_dist))
            self.move_and_update(dt = drive_time, turn_drive = 1)
            
            robot_pose = self.get_robot_pose()
            dist = math.hypot(waypoint[1] - robot_pose[1], waypoint[0] - robot_pose[0])
            print("Currently at at [{}, {}, {}]".format(robot_pose[0], robot_pose[1], robot_pose[2]))
        
    def spin_localisation(self, div = 8):
        robot_pose = self.get_robot_pose()
        total_rotation = 2*math.pi + robot_pose[2]
        
        self.set_notification('Spin-localising...')
        update_display()
        for d in reversed(range(div+1)):
            robot_pose = self.get_robot_pose()
            if d != 0:
                pose_to_rot = (total_rotation - robot_pose[2])/d
            else: 
                pose_to_rot = (total_rotation - robot_pose[2])
            pose_to_rot = clamp_angle(pose_to_rot)
            anticlockwise = np.sign(pose_to_rot)
            pose_to_rot = abs(pose_to_rot)
            turn_time = pose_to_rot/2*self.baseline/(self.scale*self.wheel_vel)

            print("Spin localising: Turning for {:.4f} seconds for {:.4f} rad".format(turn_time, anticlockwise*pose_to_rot))

            self.move_and_update(dt = turn_time, turn_drive = 0, dir=anticlockwise, force_inference = True)
            update_display()

    def rotate(self, dir, pose_to_rot=0.785398):
        pose_to_rot = clamp_angle(pose_to_rot)
        pose_to_rot = abs(pose_to_rot)
        turn_time = pose_to_rot/2*self.baseline/(self.scale*self.wheel_vel)
        
        print("Turning for {:.2f} seconds for {:.2f}rad".format(turn_time, pose_to_rot))
        self.move_and_update(dt = turn_time, turn_drive = 0, dir = dir)
    
    def forward(self, dir, dist):
        # after turning, drive straight to the waypoint
        drive_time = dist/(self.scale*self.wheel_vel)  # replace with your calculation
        print("Driving for {:.2f} seconds for {:.2f}m".format(drive_time, dist))
        self.move_and_update(dt = drive_time, turn_drive = 1, dir = dir)
    
    def move_and_update(self, dt = 1e-3, turn_drive = 0, dir = 1, force_est = False, force_inference = False):
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

                lv, rv = self.pibot.set_velocity(move, turning_tick=self.wheel_vel, time=dt)
            elif turn_drive == 1: # translating only
                if dir > 0:
                    move = [1, 0]
                else: 
                    move = [-1, 0]
                lv, rv = self.pibot.set_velocity(move, tick=self.wheel_vel, time=dt)
            elif turn_drive == 2: # turning and moving
                if dir > 0:
                    move = [1, 1]
                else:
                    move = [1, -1]
                rot_wheel_vel = abs(rot_wheel_vel)

                print("Driving for {:.2f} seconds with v = {:.2f}, w = {:.2f}".format(dt, self.wheel_vel, rot_wheel_vel))
                lv, rv = operate.pibot.set_velocity(move, tick=self.wheel_vel, turning_tick=rot_wheel_vel, time=dt)
            else:
                print('Warning: Unidentified move command')
                
            if not self.pibot.success and self.backup:
                print('Pibot command failed, backup-ing')
                self.command['output'] = True
                self.record_data()
                self.fruitPoseEst.refresh_est_pose_local()
                self.fruitPoseEst.save_targets()
                self.save_compiled_map()
                self.save_compiled_map()
        
        if turn_drive == 0 and not self.no_delay: # wait if spinning since it causes the most blur
            time.sleep(0.1)

        self.take_pic()
        
        drive_meas = Drive(lv, rv, dt)
        self.update_slam(drive_meas)

        # running detector and saving detector
        if force_inference or self.estimator_enabled:
            self.command['inference'] = True
        
        # do not save raw images during proper run
        if args.save_img == 1:
            self.command['save_image'] = True 

        self.save_image()
        num_targets, _ = self.detect_target()
        self.fruitPoseEst.label_img = self.detector_output
        self.fruitPoseEst.label_robot_pose = self.get_robot_pose()
        
        # if num_targets > 0: # only save inference if something is detected
        #     self.command['save_inference'] = True
        
        operate.record_data()
        
        if self.estimator_enabled:
            if num_targets > 0 or force_est: # only recalculate if something is detected
                operate.fruitPoseEst.refresh_est_pose_local()
            
        update_display()
        
    def create_waypoints_surface(self):
        waypoints_surface = pygame.Surface([500, 500], pygame.SRCALPHA, 32) # transparent surface
        for index, xy in enumerate(self.waypoints):
            x_im, y_im = to_im_coor(xy, [500, 500], m2pixel=156.25)
            waypoints_surface.blit(self.waypoint_pics[index%len(self.waypoint_pics)], (x_im-10, y_im-20))
            label = LABEL_FONT.render(f'{index+1}', False, (0, 0, 0))
            waypoints_surface.blit(label, (x_im-label.get_width()/4, y_im-20))
        return waypoints_surface
    
    def create_web_surface(self):
        web_surface = pygame.Surface([500, 500], pygame.SRCALPHA, 32) # transparent surface
        # loop through ekf marker to draw lines
        if self.ekf.markers.shape[1] < 2:
            return web_surface # empty surface
        
        for i in range(self.ekf.markers.shape[1]-1):
            for j in range(i+1, self.ekf.markers.shape[1]):
                from_marker = self.ekf.markers[:, i]
                to_marker = self.ekf.markers[:, j]
                from_marker = (from_marker[0].item(), from_marker[1].item())
                to_marker = (to_marker[0].item(), to_marker[1].item())
                from_marker_im = to_im_coor(from_marker, [500, 500], m2pixel=156.25)
                to_marker_im = to_im_coor(to_marker, [500, 500], m2pixel=156.25)
                
                middle_point = ((to_marker[0]+from_marker[0])/2, (to_marker[1]+from_marker[1])/2)
                angle = math.atan2(from_marker[1] - to_marker[1], from_marker[0] - to_marker[0])
                middle_point_end_1 = (middle_point[0] + 0.05*math.cos(math.pi/2+angle), middle_point[1] + 0.05*math.sin(math.pi/2+angle))
                middle_point_end_2 = (middle_point[0] - 0.05*math.cos(math.pi/2+angle), middle_point[1] - 0.05*math.sin(math.pi/2+angle))

                middle_point_end_1_im, middle_point_end_2_im = to_im_coor(middle_point_end_1, [500, 500], m2pixel=156.25), to_im_coor(middle_point_end_2, [500, 500], m2pixel=156.25)
                
                pygame.draw.line(web_surface, (100, 100, 100), from_marker_im, to_marker_im)
                pygame.draw.line(web_surface, (100, 100, 100), middle_point_end_1_im, middle_point_end_2_im)

        return web_surface
        

def update_display():
    # visualise
    operate.draw(canvas)
    
    # drawing target fruits
    fruits_surface = operate.fruitPoseEst.draw_local(res = res, m2pixel = m2pixel)
    canvas.blit(fruits_surface, map_location)
    
    # drawing waypoints:
    if len(operate.waypoints) != 0:
        waypoints_surface = operate.create_waypoints_surface()
        canvas.blit(waypoints_surface, map_location)
    
    pygame.display.update()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    parser.add_argument("--save_img", metavar='', type=int, default=0)
    parser.add_argument("--backup", metavar='', type=int, default=0)
    parser.add_argument("--load_checkpoint", metavar='', type=int, default=0)
    
    args, _ = parser.parse_known_args()
    
    if args.backup == 0:
        print('Warning: Backup disabled')
    else:
        print('Backup enabled. ')
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    LABEL_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 15)
    ALT_TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 43)
    
    width, height = 880, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2022 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading_large_phase_1.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    v_pad = 40
    h_pad = 20
    map_location = (2*h_pad+320, v_pad)
    sz = (3.2, 3.2)
    res = (500, 500)
    m2pixel = res[1]/sz[1] # 500 pixels / 3.2m

    map_area = pygame.Surface(res)
    map_area.set_alpha(0)
    canvas.blit(map_area, map_location)
    
    map_rect = map_area.get_rect().move(map_location[0], map_location[1])
    
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
                canvas.blit(pygame.image.load('pics/8bit/explosion_blue.png'), (x_-100, 565-100))
            else:
                if fire_count%10 == 0:
                    rand_x = random.randint(int(800-fire_count/10), width-10)
                    rand_y = random.randint(int(565-fire_count/10), height-10)
                    rand_int = random.randint(1, 5)
                    canvas.blit(pygame.image.load(f'pics/8bit/fire_blue_{rand_int}.png'), (rand_x, rand_y))
            pygame.display.update()
            fire_count += 1
    
    canvas.blit(splash, (0, 0))
    
    operate = Operate(args)
    if args.backup == 1:
        operate.backup = True
    if args.load_checkpoint == 1:
        print("Warning: Loading checkpoint")
        operate.ekf.load_checkpoint()

    operate.notification = 'Teleoperating'
    
    try:
        while start:
            operate.update_keyboard()
            
            operate.take_pic()
            
            drive_meas = operate.control()
            operate.update_slam(drive_meas)

            num_targets, _ = operate.detect_target()            
            # update fruitPoseEst
            operate.fruitPoseEst.label_img = operate.detector_output
            operate.fruitPoseEst.label_robot_pose = operate.get_robot_pose()
        
            operate.record_data()

            if num_targets > 0: # only recalculate if something is detected
                operate.fruitPoseEst.refresh_est_pose_local()
            
            operate.save_image()
            
            # visualise
            update_display()

    except (SystemExit):
        pygame.quit() # In the case of pygame
        print(''.join(traceback.format_tb(sys.exc_info()[2])))
        if args.backup == 1:
            print('Backup-ing')
            operate.command['output'] = True
            operate.record_data()
            operate.fruitPoseEst.refresh_est_pose_local()
            operate.fruitPoseEst.save_targets()
            operate.save_compiled_map()
        