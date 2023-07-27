# access each wheel and the camera onboard of Alphabot

import numpy as np
import requests
import cv2 
import time

class Alphabot:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.wheel_vel = [0, 0]
        self.image_shape = (640,480)
        self.session = requests.Session()
        self.success = True

    ##########################################
    # Change the robot velocity here
    # tick = forward speed
    # turning_tick = turning speed
    ########################################## 
    def set_velocity(self, command, tick=10, turning_tick=5, time=0): 
        l_vel = command[0]*tick - command[1]*turning_tick
        r_vel = command[0]*tick + command[1]*turning_tick
        self.wheel_vel = [l_vel, r_vel]
        if time == 0:
            try:
                self.session.get(
                f"http://{self.ip}:{self.port}/robot/set/velocity?value="+str(l_vel)+","+str(r_vel))
            except:
                self.success = False
                pass
        else:
            assert (time > 0), "Time must be positive."
            assert (time < 30), "Time must be less than network timeout (20s)."
            try: 
                self.session.get(
                    "http://"+self.ip+":"+str(self.port)+"/robot/set/velocity?value="+str(l_vel)+","+str(r_vel)
                                +"&time="+str(time))
            except:
                self.success = False
                pass
        return l_vel, r_vel
        
    def get_image(self):
        try:
            tick = time.time()
            r = self.session.get(f"http://{self.ip}:{self.port}/camera/get")
            print(f'Time elapsed: {time.time()-tick}')
            img = cv2.imdecode(np.frombuffer(r.content,np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Image retrieval timed out.")
            img = np.zeros((240,320,3), dtype=np.uint8)
            self.success = False
            
        return img
    
    def play_music(self):
        try:
            self.session.get(f"http://{self.ip}:{self.port}/robot/music")
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Play music timed out.")
            self.success = False
            
    def honk(self):
        try:
            self.session.get(f"http://{self.ip}:{self.port}/robot/honk")
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Honking timed out.")
            self.success = False
 
    def meow(self):
        try:
            self.session.get(f"http://{self.ip}:{self.port}/robot/meow")
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Meow timed out.")
            self.success = False
            
    def led_on(self):
        try:
            self.session.get(f"http://{self.ip}:{self.port}/robot/led/on")
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("LED timed out.")
            self.success = False
            
    def led_red(self):
        try:
            self.session.get(f"http://{self.ip}:{self.port}/robot/led/red")
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("LED timed out.")
            self.success = False
            
    def led_green(self):
        try:
            self.session.get(f"http://{self.ip}:{self.port}/robot/led/green")
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("LED timed out.")
            self.success = False
            
    def led_off(self):
        try:
            self.session.get(f"http://{self.ip}:{self.port}/robot/led/off")
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("LED timed out.")
            self.success = False

    def ir_detect(self):
        try:
            r = self.session.get(f"http://{self.ip}:{self.port}/ir/get")
            dr, dl = np.frombuffer(r.content,int)
            return dr, dl
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("IR timed out.")
            self.success = False
    
    def us_distance(self):
        try:
            return 0
            r = self.session.get(f"http://{self.ip}:{self.port}/us/get")
            dist = np.frombuffer(r.content,float)
            return dist
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Ultrasonic distance meter timed out.")
            self.success = False