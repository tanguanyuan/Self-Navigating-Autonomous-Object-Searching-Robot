# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math
from machinevisiontoolbox import Image
from scipy import stats
import matplotlib.pyplot as plt
import PIL
import pygame

class FruitPoseEst:
    def __init__(self, confirmed_fruits_list, confirmed_fruits_locations):
        self.fruits_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
        self.confirmed_fruits_list = confirmed_fruits_list
        self.confirmed_fruits_locations = confirmed_fruits_locations

        self.fruits_pics = []
        for fruit in self.fruits_list:
            f_ = f'./pics/8bit/{fruit}.png'
            self.fruits_pics.append(pygame.image.load(f_))
    
        fileK = "{}intrinsic.txt".format('./calibration/param/')
        self.camera_matrix = np.loadtxt(fileK, delimiter=',')
        self.base_dir = Path('./')

    # use the machinevision toolbox to get the bounding box of the detected target(s) in an image
    def get_bounding_box(self, target_number, image_path):
        image = PIL.Image.open(image_path).resize((640,480), PIL.Image.Resampling.NEAREST)
        target = Image(image)==target_number
        blobs = target.blobs()
        [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
        width = abs(u1-u2)
        height = abs(v1-v2)
        center = np.array(blobs[0].centroid).reshape(2,)
        box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
        # plt.imshow(fruit.image)
        # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
        # plt.show()
        # assert len(blobs) == 1, "An image should contain only one object of each target type"
        return box

    # read in the list of detection results with bounding boxes and their matching robot pose info
    def get_image_info(self, base_dir, file_path, image_poses):
        # there are at most five types of targets in each image
        target_lst_box = [[], [], [], [], []]
        target_lst_pose = [[], [], [], [], []]
        completed_img_dict = {}

        # add the bounding box info of each target in each image
        # target labels: 1 = redapple, 2 = greenapple, 3 = orange, 4 = mango, 5=capsicum, 0 = not_a_target
        img_vals = set(Image(base_dir / file_path, grey=True).image.reshape(-1))

        for target_num in img_vals:
            if target_num > 0:
                try:
                    box = self.get_bounding_box(target_num, base_dir/file_path) # [x,y,width,height]
                    pose = image_poses[file_path] # [x, y, theta]
                    target_lst_box[target_num-1].append(box) # bouncing box of target
                    target_lst_pose[target_num-1].append(np.array(pose).reshape(3,)) # robot pose
                except ZeroDivisionError:
                    pass

        # if there are more than one objects of the same type, combine them
        for i in range(5):
            if len(target_lst_box[i])>0:
                box = np.stack(target_lst_box[i], axis=1)
                pose = np.stack(target_lst_pose[i], axis=1)
                completed_img_dict[i+1] = {'target': box, 'robot': pose}
            
        return completed_img_dict
    
    # estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
    def estimate_pose(self, base_dir, camera_matrix, completed_img_dict):
        camera_matrix = camera_matrix
        focal_length = camera_matrix[0][0]
        
        # New values
        target_dimensions = []
        redapple_dimensions = [0.074, 0.074, 0.090]
        target_dimensions.append(redapple_dimensions)
        greenapple_dimensions = [0.081, 0.081, 0.080] # changed to include the stem
        target_dimensions.append(greenapple_dimensions)
        orange_dimensions = [0.075, 0.075, 0.076]
        target_dimensions.append(orange_dimensions)
        mango_dimensions = [0.113, 0.067, 0.060] # measurements when laying down
        target_dimensions.append(mango_dimensions)
        capsicum_dimensions = [0.073, 0.073, 0.100]
        target_dimensions.append(capsicum_dimensions)

        target_list = self.fruits_list

        target_pose_dict = {}
        # for each target in each detection output, estimate its pose
        for target_num in completed_img_dict.keys():
            box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
            robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
            true_height = target_dimensions[target_num-1][2]
            
            ######### Replace with your codes #########
            # TODO: compute pose of the target based on bounding box info and robot's pose
            depth = true_height/box[3]*focal_length
            u0 = camera_matrix[0][2]
            hor = -(box[0]-u0)/focal_length*depth
            dist = math.hypot(depth, hor)
            new_theta = robot_pose[2] + math.atan2(-(box[0]-u0), focal_length)
            x = robot_pose[0] + math.cos(new_theta)*dist
            y = robot_pose[1] + math.sin(new_theta)*dist
            x = x[0]
            y = y[0]
            
            # Override readings if the fruit location is given
            # if target_list[target_num-1] in self.confirmed_fruits_list:
            #     idx = self.confirmed_fruits_list.index(target_list[target_num-1])
            #     x = self.confirmed_fruits_locations[idx][0]
            #     y = self.confirmed_fruits_locations[idx][1]
            
            # filter readings if the estimation is out of bounds
            if self.out_of_range(x, y):
                continue

            target_pose = {'x': x, 'y': y}
            
            target_pose_dict[target_list[target_num-1]] = target_pose
        
        # Add and override readings if fruit location is given
        for idx, target in enumerate(self.confirmed_fruits_list):
            x = self.confirmed_fruits_locations[idx][0]
            y = self.confirmed_fruits_locations[idx][1]
            
            target_pose = {'x': x, 'y': y}
            target_pose_dict[target] = target_pose
            ###########################################

        return target_pose_dict

    # merge the estimations of the targets so that there are at most 1 estimate for each target type
    def merge_estimations(self, target_map):
        target_map = target_map
        redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = [], [], [], [], []
        target_est = {}
        num_per_target = 1 # max number of units per target type. We are only use 1 unit per fruit type
        # combine the estimations from multiple detector outputs
        for f in target_map:
            for key in target_map[f]:
                if key.startswith('redapple'):
                    redapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('greenapple'):
                    greenapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('orange'):
                    orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('mango'):
                    mango_est.append(np.array(list(target_map[f][key].values()), dtype=float))
                elif key.startswith('capsicum'):
                    capsicum_est.append(np.array(list(target_map[f][key].values()), dtype=float))

        ######### Replace with your codes #########
        # TODO: the operation below is the default solution, which simply takes the first estimation for each target type.
        # Replace it with a better merge solution.
        if len(redapple_est) > num_per_target:
            redapple_est = stats.trim_mean(redapple_est, 0.125)
            redapple_est = [redapple_est]
        if len(greenapple_est) > num_per_target:
            greenapple_est = stats.trim_mean(greenapple_est, 0.125)
            greenapple_est = [greenapple_est]
        if len(orange_est) > num_per_target:
            orange_est = stats.trim_mean(orange_est, 0.125)
            orange_est = [orange_est]
        if len(mango_est) > num_per_target:
            mango_est = stats.trim_mean(mango_est, 0.125)
            mango_est = [mango_est]
        if len(capsicum_est) > num_per_target:
            capsicum_est = stats.trim_mean(capsicum_est, 0.125)
            capsicum_est = [capsicum_est]

        for i in range(num_per_target):
            try:
                target_est['redapple_'+str(i)] = {'x':redapple_est[i][0], 'y':redapple_est[i][1]}
            except:
                pass
            try:
                target_est['greenapple_'+str(i)] = {'x':greenapple_est[i][0], 'y':greenapple_est[i][1]}
            except:
                pass
            try:
                target_est['orange_'+str(i)] = {'x':orange_est[i][0], 'y':orange_est[i][1]}
            except:
                pass
            try:
                target_est['mango_'+str(i)] = {'x':mango_est[i][0], 'y':mango_est[i][1]}
            except:
                pass
            try:
                target_est['capsicum_'+str(i)] = {'x':capsicum_est[i][0], 'y':capsicum_est[i][1]}
            except:
                pass
        ###########################################
            
        return target_est

    # def IQRmethod(self, est):
    #     for j in range(len(est)):
    #         est[j] = est[j].tolist()
    #         #for k in len(greenapple_est[j]):
    #         Q1 = np.percentile(est[j], 25, method= 'midpoint')
    #         Q3 = np.percentile(est[j], 75, method= 'midpoint')
    #         IQR = Q3 - Q1
    #         upbound = Q3+1.5*IQR
    #         lowbound = Q1-1.5*IQR

    #         leng = len(est[j])
    #         #print("j: ", j)
    #         #print("leng: ",leng)
    #         #print("redapple_est[j]: ",est[j])
    #         for i in range(leng):
    #             #print("i: ", i)
    #             #print("redapple_est[j][0]: ",redapple_est[j][0])
    #             #print("redapple_est[j][0]: ",redapple_est[j][1])
    #             #print("redapple_est[j][i]: ",redapple_est[j][i])
    #             if est[j][i] < lowbound:
    #                 est[j][i] = NULL
    #             elif est[j][i] > upbound:
    #                 est[j][i] = NULL
    #         est[j] = np.array(est[j]) 
    
    def out_of_range(self, x, y):
        max_coor = 3.2
        min_coor = -3.2
        
        if (x > max_coor or x < min_coor or y > max_coor or y < min_coor):
            return True
        else:
            return False
    
    def calc_est_pose(self):
        # a dictionary of all the saved detector outputs
        image_poses = {}
        with open(self.base_dir/'lab_output/images.txt') as fp:
            for line in fp.readlines():
                pose_dict = ast.literal_eval(line)
                image_poses[pose_dict['imgfname']] = pose_dict['pose']
        
        # estimate pose of targets in each detector output
        target_map = {}        
        for file_path in image_poses.keys():
            completed_img_dict = self.get_image_info(self.base_dir, file_path, image_poses)
            target_map[file_path] = self.estimate_pose(self.base_dir, self.camera_matrix, completed_img_dict)

        # merge the estimations of the targets so that there are only one estimate for each target type
        target_est = self.merge_estimations(target_map)
                        
        # save target pose estimations
        with open(self.base_dir/'lab_output/targets.txt', 'w') as fo:
            json.dump(target_est, fo)

    # read in the object poses
    def parse_map(self, fname: str) -> dict:
        with open(fname,'r') as f:
            gt_dict = ast.literal_eval(f.readline())        
            redapple_gt, greenapple_gt, orange_gt, mango_gt, capsicum_gt = [], [], [], [], []

            # remove unique id of targets of the same type 
            for key in gt_dict:
                if key.startswith('redapple'):
                    redapple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                elif key.startswith('greenapple'):
                    greenapple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                elif key.startswith('orange'):
                    orange_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                elif key.startswith('mango'):
                    mango_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                elif key.startswith('capsicum'):
                    capsicum_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
        # if more than 1 estimation is given for a target type, only the first estimation will be used
        num_per_target = 1 # max number of units per target type. We are only use 1 unit per fruit type
        if len(redapple_gt) > num_per_target:
            redapple_gt = redapple_gt[0:num_per_target]
        if len(greenapple_gt) > num_per_target:
            greenapple_gt = greenapple_gt[0:num_per_target]
        if len(orange_gt) > num_per_target:
            orange_gt = orange_gt[0:num_per_target]
        if len(mango_gt) > num_per_target:
            mango_gt = mango_gt[0:num_per_target]
        if len(capsicum_gt) > num_per_target:
            capsicum_gt = capsicum_gt[0:num_per_target]

        return redapple_gt, greenapple_gt, orange_gt, mango_gt, capsicum_gt

    def to_im_coor(self, xy, res, m2pixel):
        w, h = res
        x, y = xy
        x_im = int(x*m2pixel+w/2.0)
        y_im = int(-y*m2pixel+h/2.0)
        return (x_im, y_im)

    def draw(self, res = (500, 500), m2pixel = 156.25):
        redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = self.parse_map(self.base_dir/'lab_output/targets.txt')
        fruits_list = [redapple_est, greenapple_est, orange_est, mango_est, capsicum_est]
        fruits_surface = pygame.Surface(res, pygame.SRCALPHA, 32) # transparent surface
        
        for index, xy_dict in enumerate(fruits_list):
            if xy_dict:
                xy = [xy_dict[0][0], xy_dict[0][1]]
                x_im, y_im = self.to_im_coor(xy, res, m2pixel=m2pixel)
                pic = self.fruits_pics[index%len(self.fruits_pics)]
                fruits_surface.blit(pic, (x_im-pic.get_width()/2, y_im-16-pic.get_height()/2))

        return fruits_surface     
    
    def get_search_order(self):
        redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = self.parse_map(self.base_dir/'lab_output/targets.txt')
        fruits_location_list = [redapple_est, greenapple_est, orange_est, mango_est, capsicum_est]
        search_order = self.confirmed_fruits_locations.copy()
        
        # appending locations of unknown fruits
        for fruit in self.fruits_list:
            if fruit in self.confirmed_fruits_list:
                continue
            
            try: 
                idx = self.confirmed_fruits_list.index(fruit)
                search_order.append(fruits_location_list[idx])
            except:
                pass
        
        return search_order



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

if __name__ == "__main__":
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map('M4_true_map.txt')
    fruitPoseEst = FruitPoseEst(fruits_list, fruits_true_pos)
    fruitPoseEst.calc_est_pose()
    targets_list = fruitPoseEst.get_search_order()
