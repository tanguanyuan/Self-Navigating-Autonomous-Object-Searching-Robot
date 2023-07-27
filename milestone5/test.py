from auto_fruit_search import clamp_angle
import numpy as np

pose = 2*np.pi + np.pi
target_pose = 10*np.pi + np.pi/4

pose_to_rot = np.pi/4

print(clamp_angle(pose_to_rot, -2*np.pi, 0))
