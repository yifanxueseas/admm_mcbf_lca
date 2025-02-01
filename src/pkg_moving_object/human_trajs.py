#!/usr/bin/env python
import numpy as np
import time
import sys
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R

obs_num = 5
# steps = 200
# dt = 0.4 #regular
dt = 0.05 #dubin
# model_name = "star_shaped_"
# model_name = "obs"
model_name = "actor"
#1.3 or 0
default_z = 0.05

obs_position_list = np.zeros((obs_num,20,3))

#circler
k=0
obs_position_list[k,0] = [-6,-5.50, -np.pi]
obs_position_list[k,1] = [-10,-5.50, -3/2*np.pi]
obs_position_list[k,2] = [-10,-2.7, -2*np.pi]
obs_position_list[k,3] = [-6,-2.7, -3/2*np.pi]
obs_position_list[k,4] = [-6, 2.7, -3/2*np.pi]
obs_position_list[k,5] = [-6, 5, -np.pi]
obs_position_list[k,6] = [-10, 5, -1/2*np.pi]
obs_position_list[k,7] = [-10, 2, -np.pi/2]
obs_position_list[k,8] = [-10, -2, -np.pi/2]
obs_position_list[k,9] = [-10, -5.5, -np.pi/2]
obs_position_list[k,10] = [-6,-5.50, -np.pi]
obs_position_list[k,11] = [-10,-5.50, -3/2*np.pi]
obs_position_list[k,12] = [-10,-5.50, -3/2*np.pi]
# obs_position_list[k,12] = [-10, -2.5, -np.pi/2]
# obs_position_list[k,13] = [-10, -5.5, 0]
# obs_position_list[k,14] = [-6, -5.5, 0]
# obs_position_list[k,15] = [-6,-5.50, -np.pi]

k=1
obs_position_list[k,0] = [-15, 8, -np.pi/2]
obs_position_list[k,1] = [-15, 5.5, 0]
obs_position_list[k,2] = [-10.4, 5.5, 0]
obs_position_list[k,3] = [-10.4, 5.5, -np.pi/2]
obs_position_list[k,4] = [-10.2, 1.5, -np.pi/2]
obs_position_list[k,5] = [-10, -2.5, -np.pi/2]
obs_position_list[k,6] = [-10, -5.5, -np.pi/2]
obs_position_list[k,7] = [-10, -5.5, 0]
obs_position_list[k,8] = [-6, -5.5, 0]
obs_position_list[k,9] = [-6,-5.50, -np.pi]
obs_position_list[k,10] = [-10,-5.50, -np.pi]
obs_position_list[k,11] = [-10,-5.50, -3/2*np.pi]
obs_position_list[k,12] = [-10,-2, -3/2*np.pi]
obs_position_list[k,13] = [-10,-2, -2*np.pi]
obs_position_list[k,14] = [-6,-2, -2*np.pi]
obs_position_list[k,15] = [-6,-2, -3/2*np.pi]
obs_position_list[k,16] = [-6, 2, -3/2*np.pi]
# obs_position_list[k,17] = [-6, 6, -3/2*np.pi]
# obs_position_list[k,18] = [-6, 6, -3/2*np.pi]

#latest
# obs_position_list[k,0] = [-14,10.50, 0]
# obs_position_list[k,1] = [-15, 8, -np.pi/2]
# obs_position_list[k,2] = [-15, 5.7, 0]
# obs_position_list[k,3] = [-10, 5.7, 0]
# obs_position_list[k,4] = [-10, 5.7, -np.pi/2]
# obs_position_list[k,5] = [-10, 1.5, -np.pi/2]
# obs_position_list[k,6] = [-10, -2.5, -np.pi/2]
# obs_position_list[k,7] = [-10, -5.5, -np.pi/2]
# obs_position_list[k,8] = [-10, -5.5, 0]
# obs_position_list[k,9] = [-6, -5.5, 0]
# obs_position_list[k,10] = [-6,-5.50, -np.pi]
# obs_position_list[k,11] = [-10,-5.50, -np.pi]
# obs_position_list[k,12] = [-10,-5.50, -3/2*np.pi]
# obs_position_list[k,13] = [-10,-2, -3/2*np.pi]
# obs_position_list[k,14] = [-10,-2, -2*np.pi]
# obs_position_list[k,15] = [-6,-2, -2*np.pi]
# obs_position_list[k,16] = [-6,-2, -3/2*np.pi]
# obs_position_list[k,17] = [-6, 2, -3/2*np.pi]
# obs_position_list[k,18] = [-6, 6, -3/2*np.pi]
# obs_position_list[k,19] = [-6, 6, -3/2*np.pi]

#nurses
k=2
obs_position_list[k,0] = [-16,-10, np.pi/2]
obs_position_list[k,1] = [-16,-6, np.pi/2]
obs_position_list[k,2] = [-16,-2, np.pi/2]
obs_position_list[k,3] = [-16,2, np.pi/2]
obs_position_list[k,4] = [-16,5.5, np.pi/2]
obs_position_list[k,5] = [-15,5.5, np.pi/2]
obs_position_list[k,6] = [-15,9, np.pi/2]
obs_position_list[k,7] = [-15,9, np.pi/2]
obs_position_list[k,8] = [-15,9, -np.pi/2]
obs_position_list[k,9] = [-15,5.5, -np.pi/2]
obs_position_list[k,10] = [-16,5.5, -np.pi/2]
obs_position_list[k,11] = [-16,2, -np.pi/2]
obs_position_list[k,12] = [-16,-2, -np.pi/2]
obs_position_list[k,13] = [-16,-6, -np.pi/2]
obs_position_list[k,14] = [-16,-10, -np.pi/2]
obs_position_list[k,15] = [-16, -10, 0]
obs_position_list[k,16] = [-16, -10, 0]
# obs_position_list[k,17] = [-16, -10, 0]
# obs_position_list[k,18] = [-16, -10, 0]

#latest
# obs_position_list[k,0] = [-16,-10, 0]
# obs_position_list[k,1] = [-16,-10, np.pi/2]
# obs_position_list[k,2] = [-16,-6, np.pi/2]
# obs_position_list[k,3] = [-16,-2, np.pi/2]
# obs_position_list[k,4] = [-16,2, np.pi/2]
# obs_position_list[k,5] = [-16,5.5, np.pi/2]
# obs_position_list[k,6] = [-15,5.5, np.pi/2]
# obs_position_list[k,7] = [-15,9, np.pi/2]
# obs_position_list[k,8] = [-15,9, np.pi/2]
# obs_position_list[k,9] = [-15,9, -np.pi/2]
# obs_position_list[k,10] = [-15,5.5, -np.pi/2]
# obs_position_list[k,11] = [-16,5.5, -np.pi/2]
# obs_position_list[k,12] = [-16,2, -np.pi/2]
# obs_position_list[k,13] = [-16,-2, -np.pi/2]
# obs_position_list[k,14] = [-16,-6, -np.pi/2]
# obs_position_list[k,15] = [-16,-10, -np.pi/2]
# obs_position_list[k,16] = [-16, -10, 0]
# obs_position_list[k,17] = [-16, -10, 0]
# obs_position_list[k,18] = [-16, -10, 0]
# obs_position_list[k,19] = [-16, -10, 0]

# obs_position_list[0,0] = [-13,0, 0]
# obs_position_list[0,1] = [-13,0,np.pi]
# obs_position_list[0,2] = [-16,0,np.pi]
# obs_position_list[0,3] = [-16,0,3/2*np.pi]
# obs_position_list[0,4] = [-16,-4.5,3/2*np.pi]
# obs_position_list[0,5] = [-16,-9,3/2*np.pi]
# obs_position_list[0,6] = [-16,-9,1/2*np.pi]
# obs_position_list[0,7] = [-16,-4.5,1/2*np.pi]
# obs_position_list[0,8] = [-16,0,1/2*np.pi]
# obs_position_list[0,9] = [-16,0,0]
# obs_position_list[0,10] = [-16,0,0]
# obs_position_list[0,11] = [-16,0,0]
# obs_position_list[0,12] = [-16,0,0]
# obs_position_list[0,13] = [-16,0,0]
# obs_position_list[0,14] = [-16,0,0]
# obs_position_list[0,15] = [-16,0,0]
# obs_position_list[0,16] = [-16,0,0]
# obs_position_list[0,17] = [-16,0,0]
# obs_position_list[0,18] = [-16,0,0]
# obs_position_list[0,19] = [-16,0,0]

# obs_position_list[1,0] = [-14.9,-1.75, -0.76]
# obs_position_list[1,1] = [-14.9,-1.75, 0.76]
# obs_position_list[1,2] = [-13,0, 0.76]
# obs_position_list[1,3] = [-13,0, 0]
# obs_position_list[1,4] = [-13,0, 0]
# obs_position_list[1,5] = [-13,0, 0]
# obs_position_list[1,6] = [-13,0, 0]
# obs_position_list[1,7] = [-13,0, 0]
# obs_position_list[1,8] = [-13,0, 0]
# obs_position_list[1,9] = [-13,0, 0]
# obs_position_list[1,10] = [-13,0, 0]
# obs_position_list[1,11] = [-13,0, 0]
# obs_position_list[1,12] = [-13,0, 0]
# obs_position_list[1,13] = [-13,0, 0]
# obs_position_list[1,14] = [-13,0, 0]
# obs_position_list[1,15] = [-13,0, 0]
# obs_position_list[1,16] = [-13,0, 0]
# obs_position_list[1,17] = [-13,0, 0]
# obs_position_list[1,18] = [-13,0, 0]
# obs_position_list[1,19] = [-13,0, 0]

# obs_position_list[2,0] = [-15.0,2.0, 0.565]
# obs_position_list[2,1] = [-15.0,2.0, 2.4]
# obs_position_list[2,2] = [-16.2,3.5, 2.4]
# obs_position_list[2,3] = [-16.2,3.5, np.pi/2]
# obs_position_list[2,4] = [-16.2,5.7, np.pi/2]
# obs_position_list[2,5] = [-16.2,5.7, 0]
# obs_position_list[2,6] = [-12,5.7, 0]
# obs_position_list[2,7] = [-8,5.7, 0]
# obs_position_list[2,8] = [-4,5.7, 0]
# obs_position_list[2,9] = [0,5.7, 0]
# obs_position_list[2,10] = [0,5.7, 0]
# obs_position_list[2,11] = [0,5.7, 0]
# obs_position_list[2,12] = [0,5.7, 0]
# obs_position_list[2,13] = [0,5.7, 0]
# obs_position_list[2,14] = [0,5.7, 0]
# obs_position_list[2,15] = [0,5.7, 0]
# obs_position_list[2,16] = [0,5.7, 0]
# obs_position_list[2,17] = [0,5.7, 0]
# obs_position_list[2,18] = [0,5.7, 0]
# obs_position_list[2,19] = [0,5.7, 0]

#female vistor
# obs_position_list[3,0] = [-2.5,0.3, 0]
# obs_position_list[3,1] = [-2.5,0.3, -np.pi/2]
# obs_position_list[3,2] = [-6.5,0.3, -np.pi]
# obs_position_list[3,3] = [-6.5,4.3, -np.pi]
# obs_position_list[3,4] = [-6.5,5.5, -np.pi/2]
# obs_position_list[3,5] = [-8.1,5.5, -np.pi]
# obs_position_list[3,6] = [-8.1,5.5, 0]
# obs_position_list[3,7] = [-8.1,4.8, -1/2*np.pi]
# obs_position_list[3,8] = [-10.5,4.8, 0]
# obs_position_list[3,9] = [-10.5,0.8, 0]
# obs_position_list[3,10] = [-10.5,0.3, -np.pi/2]
# obs_position_list[3,11] = [-10.5,0.3, 0]
# obs_position_list[3,12] = [-10.5,-3, 0]
# obs_position_list[3,13] = [-10.5,-3, 0]
# obs_position_list[3,14] = [-10.5,-4.5, np.pi/2]
# obs_position_list[3,15] = [-6.5,-4.5, np.pi]
# obs_position_list[3,16] = [-6.5,-0.5, np.pi]
# obs_position_list[3,17] = [-6.5,3.5, np.pi]
# obs_position_list[3,18] = [-6.5,5.5, -np.pi/2]

obs_position_list[3,0] = [-2.5,0.3, -np.pi/2]
obs_position_list[3,1] = [-6.5,0.4, -np.pi/2]
obs_position_list[3,2] = [-11,0.4, -np.pi/2]
obs_position_list[3,3] = [-11,0.4, -np.pi]
obs_position_list[3,4] = [-11,2.6, -np.pi]
obs_position_list[3,5] = [-11,4.8, -3*np.pi/2]
obs_position_list[3,6] = [-8.1,4.8, -3*np.pi/2]
obs_position_list[3,7] = [-8.1,5.5, -np.pi]
obs_position_list[3,8] = [-8.1,5.5, -np.pi]
obs_position_list[3,9] = [-8.1,5.5, -3*np.pi/2]
obs_position_list[3,10] = [-5,5.5, -3/2*np.pi]
obs_position_list[3,11] = [-5,5.5, -2*np.pi]
obs_position_list[3,12] = [-5,1.5, -2*np.pi]
obs_position_list[3,13] = [-5,0.4, -5/2*np.pi]
obs_position_list[3,14] = [-9.5,0.4, -5/2*np.pi]
obs_position_list[3,15] = [-9.5,0.4, -5/2*np.pi]
obs_position_list[3,16] = [-10.5,0.4, -5/2*np.pi]
obs_position_list[3,17] = [-10.5,0.4, -5/2*np.pi]

#male vistor
k=4
obs_position_list[k,0] = [-4.5,-5.50, -3/2*np.pi]
obs_position_list[k,1] = [-4.5,-1.50, -3/2*np.pi]
obs_position_list[k,2] = [-4.5,2.50, -3/2*np.pi]
obs_position_list[k,3] = [-6,6, -3/2*np.pi]
obs_position_list[k,4] = [-6, 9, -3/2*np.pi]
obs_position_list[k,5] = [-6, 9, -3/2*np.pi]
obs_position_list[k,6] = [-6, 9, -3/2*np.pi]
# obs_position_list[k,8] = [-10, 2, -np.pi/2]
# obs_position_list[k,9] = [-10, -2, -np.pi/2]
# obs_position_list[k,10] = [-10, -5.5, -np.pi/2]
# obs_position_list[k,11] = [-6,-5.50, -np.pi]
# obs_position_list[k,12] = [-10,-5.50, -3/2*np.pi]


# obs_position_list[4,0] = [-3.3,3.3, 0]
# obs_position_list[4,1] = [-3.3,3.3, 0]
# obs_position_list[4,2] = [-3.3,3.3, -np.pi/2]
# obs_position_list[4,3] = [-3.3,0.3, -np.pi/2]
# obs_position_list[4,4] = [-3.3,0.3, -np.pi]
# obs_position_list[4,5] = [-7.3,0.5, -np.pi]
# obs_position_list[4,6] = [-9.5,0.5, -np.pi]
# obs_position_list[4,7] = [-10.5,0.5, -np.pi]
# obs_position_list[4,8] = [-10.5,0.5, -np.pi]
# obs_position_list[4,9] = [-10.5,0.5, -np.pi]
# obs_position_list[4,10] = [-10.5,0.5, -np.pi/2]
# obs_position_list[4,11] = [-10.5,-4.5, -np.pi/2]
# obs_position_list[4,12] = [-10.5,-4.5, 0]
# obs_position_list[4,14] = [-6.5,-4.5, 0]
# obs_position_list[4,15] = [-5,-4.5, np.pi/2]
# obs_position_list[4,16] = [-5,-0.5, np.pi/2]
# obs_position_list[4,17] = [-5,0.3, np.pi]
# obs_position_list[4,18] = [-9.5,0.3, np.pi]
# obs_position_list[4,19] = [-9.5,0.3, np.pi]



def x_c(iter, steps=200):
	#steps: how many iterations the human take to go from position at time t to that at time t+1
	t = int(iter/steps)
	vxy = dx_c(iter)
	x_center = obs_position_list[:,t,:]+dt*vxy*(iter % steps)
	return x_center

def dx_c(iter,steps=200):
	#steps: how many iterations the human take to go from position at time t to that at time t+1
	t = int(iter/steps)
	linear_v = (obs_position_list[:,t+1,:]-obs_position_list[:,t,:])/(dt*steps)
	return linear_v


	
