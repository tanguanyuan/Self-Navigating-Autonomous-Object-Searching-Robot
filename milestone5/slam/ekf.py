from tkinter import Y
import numpy as np
from mapping_utils import MappingUtils
import cv2
import math
import pygame
import json
import ast 

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

class EKF:
    # Implementation of an EKF for SLAM
    # The state is ordered as [x; y; theta; l1x; l1y; ...; lnx; lny]

    ##########################################
    # Utility
    # Add outlier rejection here
    ##########################################

    def __init__(self, robot, auto=False):
        # State components
        self.robot = robot
        self.markers = np.zeros((2,0))
        self.taglist = []
        
        # TODO: Modify these parameters by trial and error
        # success if the position of the robot in the arena and pos of robot in GUI matched after a few movements
        
        # Covariance matrix
        self.P = 0.01*np.eye(3) #Default: 0.1 # initial covariance of position [x, y, theta] # maybe need to set to a non-zero value 
        self.init_lm_cov = 2e3 # Default 1e3 # initial covariance of unknown, detected landmarks, will be squared when intialised
        self.init_known_lm_cov = 0 #Default: 0.01 # initial covariance of given landmarks, will be squared when intialised
        
        # cumulatives variances of R and Q, the idea is to keep it on edge so that it will correct its states
        if auto: # for phase 2 
            self.R0 = 0.05**2 #0.01**2 # theoretical 0.0025 #Default: 0.01 # minimum/cumulative covariance of measurement # default 0.01 
            self.Q0_xy = 0.00061
            self.Q0_theta = 0.27415 # 30° error            
        else:   # for phase 1
            self.R0 = 0.05**2 # theoretical 0.0025 #Default: 0.01 # minimum/cumulative covariance of measurement # default 0.01 
            self.Q0_xy = 0.0006
            self.Q0_theta = 0.0685 #0.0685 # 15° error
        #################################################
        self.robot_init_state = None
        self.lm_pics = []
        
        for i in range(1, 11):
            f_ = f'./pics/8bit/lm_{i}.png'
            self.lm_pics.append(pygame.image.load(f_))
        f_ = f'./pics/8bit/lm_unknown.png'
        self.lm_pics.append(pygame.image.load(f_))
        self.pibot_pic = pygame.image.load(f'./pics/8bit/pibot_top_flip_centered.png')
        
    def reset(self):
        self.robot.state = np.zeros((3, 1))
        self.markers = np.zeros((2,0))
        self.taglist = []  

        # Covariance matrix
        self.P = 0.01*np.eye(3)
        self.robot_init_state = None

    def load_checkpoint(self):
        with open('lab_output/slam.txt', 'r') as f:
            try:
                usr_dict = json.load(f)                   
            except ValueError as e:
                with open('lab_output/slam.txt', 'r') as f:
                    usr_dict = ast.literal_eval(f.readline()) 
        
        # reset variables
        self.markers = np.zeros((2,0))
        self.taglist = []
        self.P = 0.01*np.eye(3)
        for (i, tag) in enumerate(usr_dict["taglist"]):
            self.taglist.append(tag)
            marker_pose = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
            self.markers = np.concatenate((self.markers, marker_pose), axis=1)
            self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
            self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)

        self.P[3:, 3:] = np.array(usr_dict["covariance"])
        
    def number_landmarks(self):
        return int(self.markers.shape[1])

    def get_state_vector(self):
        state = np.concatenate(
            (self.robot.state, np.reshape(self.markers, (-1,1), order='F')), axis=0)
        return state
    
    def set_state_vector(self, state):
        self.robot.state = state[0:3,:]
        self.markers = np.reshape(state[3:,:], (2,-1), order='F')
    
    def save_map(self, fname="slam_map.txt"):
        if self.number_landmarks() > 0:
            utils = MappingUtils(self.markers, self.P[3:,3:], self.taglist)
            utils.save(fname)

    def recover_from_pause(self, measurements):
        if not measurements:
            return False
        else:
            lm_new = np.zeros((2,0))
            lm_prev = np.zeros((2,0))
            tag = []
            for lm in measurements:
                if lm.tag in self.taglist:
                    lm_new = np.concatenate((lm_new, lm.position), axis=1)
                    tag.append(int(lm.tag))
                    lm_idx = self.taglist.index(lm.tag)
                    lm_prev = np.concatenate((lm_prev,self.markers[:,lm_idx].reshape(2, 1)), axis=1)
            if int(lm_new.shape[1]) > 2:
                R,t = self.umeyama(lm_new, lm_prev)
                theta = math.atan2(R[1][0], R[0][0])
                self.robot.state[:2]=t[:2]
                self.robot.state[2]=theta
                return True
            else:
                return False
        
    ##########################################
    # EKF functions
    # Tune your SLAM algorithm here
    # ########################################

    # the prediction step of EKF
    def predict(self, raw_drive_meas):
        F = self.state_transition(raw_drive_meas)
        x = self.get_state_vector()

        # TODO: add your codes here to compute the predicted x
        self.robot.drive(raw_drive_meas)
        x_hat = x
        x_hat[0:3] = np.array([self.robot.state[0], self.robot.state[1], self.robot.state[2]]).reshape((3,1))
        self.set_state_vector(x_hat)
        
        Q = self.predict_covariance(raw_drive_meas)
        self.P = F @ self.P @ F.T + Q

    # the update step of EKF
    def update(self, measurements):
        if not measurements:
            return

        # Construct measurement index list
        tags = [lm.tag for lm in measurements]
        idx_list = [self.taglist.index(tag) for tag in tags]

        # Stack measurements and set covariance
        z = np.concatenate([lm.position.reshape(-1,1) for lm in measurements], axis=0)
        R = np.zeros((2*len(measurements),2*len(measurements)))
        for i in range(len(measurements)):
            R[2*i:2*i+2,2*i:2*i+2] = measurements[i].covariance + self.R0*np.eye(2) # suspicious 

        # Compute own measurements
        z_hat = self.robot.measure(self.markers, idx_list)
        z_hat = z_hat.reshape((-1,1),order="F")
        H = self.robot.derivative_measure(self.markers, idx_list)

        x = self.get_state_vector()

        # TODO: add your codes here to compute the updated x
        # compute Kalman Gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        # Correct state
        y = z - z_hat
        x = x + K @ y
        x[2] = clamp_angle(x[2])
        self.set_state_vector(x)

        # Correct covariance
        self.P = (np.eye(x.shape[0]) - K @ H) @ self.P

    def state_transition(self, raw_drive_meas):
        n = self.number_landmarks()*2 + 3
        F = np.eye(n)
        F[0:3,0:3] = self.robot.derivative_drive(raw_drive_meas)
        return F
    
    def predict_covariance(self, raw_drive_meas):
        n = self.number_landmarks()*2 + 3
        Q = np.zeros((n,n))
        Q0 = self.Q0_xy*np.eye(3)
        Q0[-1, -1] = self.Q0_theta
        Q[0:3,0:3] = self.robot.covariance_drive(raw_drive_meas) + Q0#+ self.Q0*np.eye(3)
        return Q

    def add_landmarks(self, measurements):
        if not measurements:
            return

        th = self.robot.state[2]
        robot_xy = self.robot.state[0:2,:]
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])

        # Add new landmarks to the state
        for lm in measurements:
            if lm.tag in self.taglist:
                # ignore known tags
                continue
                
            lm_bff = lm.position
            lm_inertial = robot_xy + R_theta @ lm_bff

            self.taglist.append(int(lm.tag))
            self.markers = np.concatenate((self.markers, lm_inertial), axis=1)

            # Create a simple, large covariance to be fixed by the update step
            self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
            self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)
            self.P[-2,-2] = self.init_lm_cov**2
            self.P[-1,-1] = self.init_lm_cov**2

    def populate_landmarks(self, positions):
        self.markers = np.zeros((2,0))
        self.taglist = []
        for i in range(positions.shape[0]):
            self.taglist.append(int(i+1))
            position = np.zeros((2, 1))
            position[0] = positions[i][0]
            position[1] = positions[i][1]
            self.markers = np.concatenate((self.markers, position), axis = 1)
            self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
            self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)
            self.P[-2,-2] = self.init_known_lm_cov**2
            self.P[-1,-1] = self.init_known_lm_cov**2
    ##########################################
    ##########################################
    ##########################################
    
    def realign_map(self):
        th = self.robot.state[2]
        robot_xy = self.robot.state[0:2,:]
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        
        # realign markers
        aligned_markers = np.zeros((2,0))
        for i in range(self.markers.shape[1]):
            lm_old = self.markers[:, i].reshape(2, 1)
            lm_aligned = np.linalg.inv(R_theta)@(lm_old - robot_xy)
            aligned_markers = np.concatenate((aligned_markers, lm_aligned), axis=1)
        
        self.markers = aligned_markers
        
        # realign robot # suppose to be 0 for xy and multiple of 2pi for theta
        self.robot.state[0:2, :] = self.robot.state[0:2, :] - robot_xy
        self.robot.state[2, :] = self.robot.state[2, :] - th
    
    def lock_markers(self):
        self.P[3:, :] *= 0.01
        self.P[:, 3:] *= 0.01
        
    def lock_one_marker(self, tag):
        try:
            i = self.taglist.index(tag)
            self.P[3+2*i:3+2*(i+1), :] *= 0.01
            self.P[:, 3+2*i:3+2*(i+1)] *= 0.01
            return f'Locked marker {tag}'
        except ValueError:
            return f'Unknown marker {tag}'
    
    def unlock_markers(self):
        self.P[3:, :] *= 100
        self.P[:, 3:] *= 100
        
    def unlock_one_marker(self, tag):
        try: 
            i = self.taglist.index(tag)
        except ValueError:
            return f'Unknown marker {tag}'
        
        self.P[3+2*i:3+2*(i+1), :] *= 100
        self.P[:, 3+2*i:3+2*(i+1)] *= 100
        return f'Unlocked marker {tag}'

    def delete_one_marker(self, tag):
        try: 
            i = self.taglist.index(tag)
        except ValueError:
            return f'Unknown marker {tag}'
        markers_ind_rem = [j for j in range(len(self.taglist)) if j != i]
        self.markers = self.markers[:, markers_ind_rem]
        self.P = np.delete(self.P, 3+2*i+1, 0)
        self.P = np.delete(self.P, 3+2*i, 0)
        self.P = np.delete(self.P, 3+2*i+1, 1)
        self.P = np.delete(self.P, 3+2*i, 1)
        self.taglist.pop(i)
        return f'Deleted marker {tag}'

    @staticmethod
    def umeyama(from_points, to_points):

        assert len(from_points.shape) == 2, \
            "from_points must be a m x n array"
        assert from_points.shape == to_points.shape, \
            "from_points and to_points must have the same shape"
        
        N = from_points.shape[1]
        m = 2
        
        mean_from = from_points.mean(axis = 1).reshape((2,1))
        mean_to = to_points.mean(axis = 1).reshape((2,1))
        
        delta_from = from_points - mean_from # N x m
        delta_to = to_points - mean_to       # N x m
        
        cov_matrix = delta_to @ delta_from.T / N
        
        U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
        cov_rank = np.linalg.matrix_rank(cov_matrix)
        S = np.eye(m)
        
        if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
            S[m-1, m-1] = -1
        elif cov_rank < m-1:
            raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
        
        R = U.dot(S).dot(V_t)
        t = mean_to - R.dot(mean_from)
    
        return R, t

    # Plotting functions
    # ------------------
    @ staticmethod
    def to_im_coor(xy, res, m2pixel):
        w, h = res
        x, y = xy
        x_im = int(x*m2pixel+w/2.0)
        y_im = int(-y*m2pixel+h/2.0)
        return (x_im, y_im)

    def draw_slam_state(self, res = (320, 500), not_pause=True):
        # Draw landmarks
        m2pixel = 100
        if not_pause:
            bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
        else:
            bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)
        # in meters, 
        lms_xy = self.markers[:2, :]
        robot_xy = self.robot.state[:2, 0].reshape((2, 1))
        lms_xy = lms_xy - robot_xy
        robot_xy = robot_xy*0
        robot_theta = self.robot.state[2,0]
        # plot robot
        start_point_uv = self.to_im_coor((0, 0), res, m2pixel)
        
        p_robot = self.P[0:2,0:2]
        axes_len,angle = self.make_ellipse(p_robot)
        canvas = cv2.ellipse(canvas, start_point_uv, 
                    (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                    angle, 0, 360, (0, 30, 56), 1)
        # draw landmards
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                # plot covariance
                Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
                axes_len, angle = self.make_ellipse(Plmi)
                canvas = cv2.ellipse(canvas, coor_, 
                    (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                    angle, 0, 360, (244, 69, 96), 1)

        surface = pygame.surfarray.make_surface(np.rot90(canvas))
        surface = pygame.transform.flip(surface, True, False)
        surface.blit(self.rot_center(self.pibot_pic, robot_theta*57.3),
                    (start_point_uv[0]-15, start_point_uv[1]-15))
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                try:
                    surface.blit(self.lm_pics[self.taglist[i]-1],
                    (coor_[0]-5, coor_[1]-5))
                except IndexError:
                    surface.blit(self.lm_pics[-1],
                    (coor_[0]-5, coor_[1]-5))
        return surface

    def draw_slam_state_world(self, res = (500, 500), not_pause=True):
        # Draw landmarks
        m2pixel = 156.25
        if not_pause:
            bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
        else:
            bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)
        canvas = cv2.imread('pics/grid.png') # add grid instead

        # in meters, 
        lms_xy = self.markers[:2, :]
        robot_xy = self.robot.state[:2, 0].reshape((2, 1))

        robot_theta = self.robot.state[2,0]
        # plot robot
        start_point_uv = self.to_im_coor((robot_xy[0], robot_xy[1]), res, m2pixel)
        p_robot = self.P[0:2,0:2]
        axes_len,angle = self.make_ellipse(p_robot)
        canvas = cv2.ellipse(canvas, start_point_uv, 
                    (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                    angle, 0, 360, (0, 30, 56), 1)
        
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                # plot covariance
                Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
                axes_len, angle = self.make_ellipse(Plmi)
                canvas = cv2.ellipse(canvas, coor_, 
                    (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                    angle, 0, 360, (244, 69, 96), 1)
        
        surface = pygame.surfarray.make_surface(np.rot90(canvas))
        surface = pygame.transform.flip(surface, True, False)
        
        # draw landmards
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                try:
                    surface.blit(self.lm_pics[self.taglist[i]-1],
                    (coor_[0]-8, coor_[1]-8))
                except IndexError:
                    surface.blit(self.lm_pics[-1],
                    (coor_[0]-8, coor_[1]-8))
                
        surface.blit(self.rot_center(self.pibot_pic, robot_theta*57.3),
                    (start_point_uv[0]-15, start_point_uv[1]-15))
        return surface

    @staticmethod
    def rot_center(image, angle):
        """rotate an image while keeping its center and size"""
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image       

    @staticmethod
    def make_ellipse(P):
        e_vals, e_vecs = np.linalg.eig(P)
        idx = e_vals.argsort()[::-1]   
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        alpha = np.sqrt(4.605)
        axes_len = e_vals*2*alpha
        if abs(e_vecs[1, 0]) > 1e-3:
            angle = np.arctan(e_vecs[0, 0]/e_vecs[1, 0])
        else:
            angle = 0
        return (axes_len[0], axes_len[1]), angle

 