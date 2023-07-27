# estimate the pose of a target object detected
from multiprocessing.sharedctypes import Value
from operator import truediv
import numpy as np
import json
from pathlib import Path
import ast
import math
from machinevisiontoolbox import Image
from scipy import stats
import PIL
import pygame
from util.helper import read_true_map, read_search_list, to_im_coor

class FruitPoseEst:
    def __init__(self, confirmed_fruits_list = [], confirmed_fruits_locations = [], search_list = []):
        self.fruits_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
        self.confirmed_fruits_list = confirmed_fruits_list
        self.confirmed_fruits_locations = confirmed_fruits_locations
        self.unconfirmed_fruits_list = [fruit for fruit in self.fruits_list if fruit not in self.confirmed_fruits_list]
        self.unconfirmed_fruits_idx_list = [self.fruits_list.index(fruit)+1 for fruit in self.unconfirmed_fruits_list]
        self.search_list = search_list
        
        self.fruits_pics = []
        for fruit in self.fruits_list:
            f_ = f'./pics/8bit/fruit_{fruit}_centered.png'
            self.fruits_pics.append(pygame.image.load(f_))
            
        self.fruits_locked_pics = []
        for fruit in self.fruits_list:
            f_ = f'./pics/8bit/fruit_{fruit}_checked.png'
            self.fruits_locked_pics.append(pygame.image.load(f_))
    
        fileK = "{}intrinsic.txt".format('./calibration/param/')
        self.camera_matrix = np.loadtxt(fileK, delimiter=',')
        self.base_dir = Path('./')
        
        self.estimated_poses = {}
        self.label_img = np.zeros((480, 640))
        self.label_robot_pose = [0, 0, 0]
        
        for fruit in self.fruits_list:
            self.estimated_poses[fruit] = np.zeros((0, 2))
        self.fruits_location_list = []
        
        self.refresh_est_pose_local()

    def erase_all(self):
        self.estimated_poses = {}
        for fruit in self.fruits_list:
            self.estimated_poses[fruit] = np.zeros((0, 2))
        self.fruits_location_list = []
        
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
            
            depth = true_height/box[3]*focal_length
            u0 = camera_matrix[0][2]
            hor = -(box[0]-u0)/focal_length*depth
            dist = math.hypot(depth, hor)
            new_theta = robot_pose[2] + math.atan2(-(box[0]-u0), focal_length)
            x = robot_pose[0] + math.cos(new_theta)*dist
            y = robot_pose[1] + math.sin(new_theta)*dist
            x = x[0]
            y = y[0]
            
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
        
        # add and override known fruits
        for idx, target in enumerate(self.confirmed_fruits_list):
            x = self.confirmed_fruits_locations[idx][0]
            y = self.confirmed_fruits_locations[idx][1]
            
            target_pose = {'x': x, 'y': y}
            target_est[f'{target}_0'] = target_pose
        
        # save target pose estimations
        with open(self.base_dir/'lab_output/targets.txt', 'w') as fo:
            json.dump(target_est, fo)

    def get_search_order(self):
        redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = self.parse_map(self.base_dir/'lab_output/targets.txt')
        fruits_location_list = [redapple_est, greenapple_est, orange_est, mango_est, capsicum_est]
        search_fruits = []
        search_order = np.zeros((0, 2))
        for fruit in self.search_list:
            idx = self.fruits_list.index(fruit)
            search_order = np.append(search_order, fruits_location_list[idx], axis = 0)
            search_fruits.append(fruit)

        # appending locations of unknown fruits
        for fruit in self.fruits_list:
            if fruit in self.search_list:
                continue
            
            try: 
                idx = self.fruits_list.index(fruit)
                search_order = np.append(search_order, fruits_location_list[idx], axis = 0)
                search_fruits.append(fruit)
            except:
                pass
        
        return search_order, search_fruits
    
    # use the machinevision toolbox to get the bounding box of the detected target(s) in an image
    def get_bounding_box_local(self, target_number):
        image = PIL.Image.fromarray(self.label_img)
        target = Image(image)==target_number
        blobs = target.blobs()
        [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
        width = abs(u1-u2)
        height = abs(v1-v2)
        center = np.array(blobs[0].centroid).reshape(2,)
        box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]

        return box    
    
     # read in the list of detection results with bounding boxes and their matching robot pose info
    def get_image_info_local(self):
        # there are at most five types of targets in each image
        target_lst_box = [[], [], [], [], []]
        target_lst_pose = [[], [], [], [], []]
        completed_img_dict = {}

        # add the bounding box info of each target in each image
        # target labels: 1 = redapple, 2 = greenapple, 3 = orange, 4 = mango, 5=capsicum, 0 = not_a_target
        img_vals = np.unique(self.label_img)

        for target_num in img_vals:
            if target_num > 0:
                try:
                    box = self.get_bounding_box_local(target_num) # [x,y,width,height]
                    pose = self.label_robot_pose # [x, y, theta]
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

    def realign_map(self, cur_robot_pose):
        th = cur_robot_pose[2]
        robot_xy = np.array(cur_robot_pose[0:2]).reshape(2, 1)
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        
        for fruit in self.fruits_list:
            for idx in range(self.estimated_poses[fruit].shape[0]):
                prev_pose =  self.estimated_poses[fruit][idx, :].reshape(2, 1)
                aligned_pose = np.linalg.inv(R_theta)@(prev_pose - robot_xy)
                self.estimated_poses[fruit][idx, :] = aligned_pose.reshape(1, 2)
        
    def refresh_est_pose_local(self):
        completed_img_dict = self.get_image_info_local()

        target_pose_dict = self.estimate_pose(self.base_dir, self.camera_matrix, completed_img_dict)

        for fruit in target_pose_dict:
            fruit_pose = [[target_pose_dict[fruit]['x'], target_pose_dict[fruit]['y']]]
            self.estimated_poses[fruit] = np.concatenate((self.estimated_poses[fruit], fruit_pose), axis = 0)

    def save_targets(self):
        redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = self.merge_estimations_local()
        
        target_est = {}
        i = 0
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

        # add and override known fruits
        for idx, target in enumerate(self.confirmed_fruits_list):
            x = self.confirmed_fruits_locations[idx][0]
            y = self.confirmed_fruits_locations[idx][1]
            
            target_pose = {'x': x, 'y': y}
            target_est[f'{target}_0'] = target_pose
        
        # save target pose estimations
        with open(self.base_dir/'lab_output/targets.txt', 'w') as fo:
            json.dump(target_est, fo, indent=2)
            
    # merge the estimations of the targets so that there are at most 1 estimate for each target type
    def merge_estimations_local(self):
        num_per_target = 1 # max number of units per target type. We are only use 1 unit per fruit type
        redapple_est = self.estimated_poses['redapple']
        greenapple_est = self.estimated_poses['greenapple']
        orange_est = self.estimated_poses['orange']
        mango_est = self.estimated_poses['mango']
        capsicum_est = self.estimated_poses['capsicum']

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
        
        return redapple_est, greenapple_est, orange_est, mango_est, capsicum_est

    def get_search_order_local(self):
        redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = self.merge_estimations_local()
        
        fruits_location_list = [redapple_est, greenapple_est, orange_est, mango_est, capsicum_est]
        search_fruits = []
        search_order = np.zeros((0, 2))
        for fruit in self.search_list:
            idx = self.fruits_list.index(fruit)
            search_order = np.append(search_order, fruits_location_list[idx], axis = 0)
            search_fruits.append(fruit)
            
        # appending locations of unknown fruits
        for fruit in self.fruits_list:
            if fruit in self.search_list:
                continue
            
            try: 
                idx = self.fruits_list.index(fruit)
                if len(fruits_location_list[idx]) == 0:
                    continue
                search_order = np.append(search_order, fruits_location_list[idx], axis = 0)
                search_fruits.append(fruit)
            except:
                pass

        return search_order, search_fruits

    def lock_fruit(self, fruit):
        fruits_location, fruits = self.get_search_order_local()
        try: 
            idx = fruits.index(fruit)
            [x, y] = fruits_location[idx, :]
        except (ValueError, IndexError):
            return f'Unknown target {fruit}' # fruit not found yet
        if fruit not in self.confirmed_fruits_list:
            self.confirmed_fruits_list.append(fruit)

            self.confirmed_fruits_locations.append([x, y])
        else:
            return f'Target {fruit} already locked'
        return f'Locked target {fruit}'
    
    def unlock_fruit(self, fruit):
        _, fruits = self.get_search_order_local()
        if fruit not in fruits:
            return f'Unknown target {fruit}' # fruit not found yet

        if fruit not in self.confirmed_fruits_list:
            return f'Target {fruit} already unlocked'

        idx = self.confirmed_fruits_list.index(fruit)
        self.confirmed_fruits_list.pop(idx)
        self.confirmed_fruits_locations.pop(idx)
            
        return f'Unlocked target {fruit}'
    
    def delete_fruit(self, fruit):
        _, fruits = self.get_search_order_local()
        if fruit not in fruits:
            return f'Unknown target {fruit}' # fruit not found yet

        self.estimated_poses[fruit] = np.zeros((0, 2))
        if fruit in self.confirmed_fruits_list:
            idx = self.confirmed_fruits_list.index(fruit)
            self.confirmed_fruits_list.pop(idx)
            self.confirmed_fruits_locations.pop(idx)
            
        return f'Deleted target {fruit}'
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

    def draw(self, res = (500, 500), m2pixel = 156.25):
        redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = self.parse_map(self.base_dir/'lab_output/targets.txt')
        fruits_list = [redapple_est, greenapple_est, orange_est, mango_est, capsicum_est]
        fruits_surface = pygame.Surface(res, pygame.SRCALPHA, 32) # transparent surface
        
        for index, xy_dict in enumerate(fruits_list):
            if xy_dict:
                xy = [xy_dict[0][0], xy_dict[0][1]]
                x_im, y_im = to_im_coor(xy, res, m2pixel=m2pixel)
                fruit = self.fruits_list[index]
                if fruit in self.confirmed_fruits_list:
                    pic = self.fruits_locked_pics[index%len(self.fruits_pics)]
                else:
                    pic = self.fruits_pics[index%len(self.fruits_pics)]
                fruits_surface.blit(pic, (x_im-pic.get_width()/2, y_im-pic.get_height()/2))

        return fruits_surface     
    
    def draw_local(self, res = (500, 500), m2pixel = 156.25):
        redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = self.merge_estimations_local()
        fruits_list = [redapple_est, greenapple_est, orange_est, mango_est, capsicum_est]
        fruits_surface = pygame.Surface(res, pygame.SRCALPHA, 32) # transparent surface
        
        for index, xy_dict in enumerate(fruits_list):
            if len(xy_dict) != 0:
                xy = [xy_dict[0][0], xy_dict[0][1]]
                x_im, y_im = to_im_coor(xy, res, m2pixel=m2pixel)
                fruit = self.fruits_list[index]
                if fruit in self.confirmed_fruits_list:
                    pic = self.fruits_locked_pics[index%len(self.fruits_pics)]
                else:
                    pic = self.fruits_pics[index%len(self.fruits_pics)]
                fruits_surface.blit(pic, (x_im-pic.get_width()/2, y_im-pic.get_height()/2))

        return fruits_surface     

if __name__ == "__main__":
    img = PIL.Image.open('lab_output/pred_0.png')
    img = np.asarray(img)

    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map('M4_true_map.txt')
    search_list = read_search_list()
    fruitPoseEst = FruitPoseEst(fruits_list, fruits_true_pos, search_list)
    fruitPoseEst.label_img = img
    fruitPoseEst.label_robot_pose = [0, 0, 0]
    fruitPoseEst.calc_est_pose()
    targets_list = fruitPoseEst.get_search_order()
    print(targets_list)
    fruitPoseEst.refresh_est_pose_local()
    search_order = fruitPoseEst.get_search_order_local()
    
    print(search_order)