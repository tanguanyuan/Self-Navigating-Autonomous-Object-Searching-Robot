!!!!!!!!! USE ETHERNET

command:
PHASE I:
python operate.py --ip (ip)
- save Aruco map: Y
- to detect targets and estimate pose: Q->F (N is depreciated)
- do both of the above and save compiled map: C

SLAM evaluation:
python SLAM_eval.py TRUEMAP.txt lab_output/slam.txt

Targets evaluation:
python CV_eval.py TRUEMAP.txt lab_output/targets.txt

PHASE II:
python auto_fruit_search.py --ip (the ip listed in hotspot settings) --port 8000 --level (1 or 2) 
	--save_img (0 or 1) --return_origin (0 or 1)

level 1: semi-auto, select waypoints by clicking on map (mainly for debugging purposes)
level 2: fully-auto, waypoints are automatically generated using A* algorithm

save_img: 1 (default) -> save raw images 0 -> do not save raw images
return_origin: 1 (default) -> go back to origin after detecting 1 target 0 -> do not go back
waypoint_inc: 6 (default) -> steps of waypoints to skip (3:10cm)