from auto_fruit_search import clamp_angle
import numpy as np

pose = 2*np.pi + np.pi
target_pose = 10*np.pi + np.pi/4

pose_to_rot = target_pose - pose
print(pose_to_rot)

print(clamp_angle(pose_to_rot))
