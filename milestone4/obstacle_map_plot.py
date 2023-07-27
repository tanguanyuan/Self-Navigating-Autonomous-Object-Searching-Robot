import json
import numpy as np
import cv2
from AStar import AStar
from auto_fruit_search import to_im_coor, clamp, to_m
import pygame 
import math 

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

astar_obstacle_file_path = "astar_obstacles.json"
with open(astar_obstacle_file_path, 'r') as j:
    contents = json.loads(j.read())
data = contents["data"]

res_a_star = (100, 100)
canvas = pygame.display.set_mode((500, 500))
sz = (3.2, 3.2)
obstacle_map = 255*np.ones(res_a_star).astype(np.uint8)

for [x, y] in data:
    obstacle_map[y, x] = 0

obstacle_ratio = 5

robot_pose = [0, 0]
target_pose = [0.4, 0.4]

start_point = to_im_coor((robot_pose[0], robot_pose[1]), res_a_star, res_a_star[1]/sz[1])
start_point = [start_point[0], start_point[1]]
target_point = find_closest_at_target([target_pose[0], target_pose[1]], [robot_pose[0], robot_pose[1]], 0.1100)
print(target_point)
end_point = to_im_coor((target_point[0], target_point[1]), res_a_star, res_a_star[1]/sz[1])
end_point = [end_point[0], end_point[1]]

astar = AStar(res_a_star[0], res_a_star[1], start_point, end_point, obstacle_ratio=obstacle_ratio, obstacle_list = data)
path = astar.main()
print(len(path))
for i in range(len(path)-1):
    obstacle_map = cv2.line(obstacle_map, (path[i].x, path[i].y), (path[i+1].x, path[i+1].y), color = (127, 127, 127), thickness = 1)

obstacle_map = cv2.line(obstacle_map, (start_point[0], start_point[1]), (end_point[0], end_point[1]), color = (200, 200, 200), thickness = 1)

cv2.namedWindow('map', cv2.WINDOW_NORMAL)
cv2.imshow('map', obstacle_map)
cv2.resizeWindow('map', 500,500)
cv2.waitKey(0)
# waypoint_idx = 0
# print(path)
# while waypoint_idx < len(path)-1:
#     waypoint_idx += 3
#     waypoint_idx = clamp(waypoint_idx, len(path)-1, 0)
#     print(waypoint_idx)