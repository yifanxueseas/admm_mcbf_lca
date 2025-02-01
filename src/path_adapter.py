import numpy as np
from matplotlib.axes import Axes # type: ignore
from path_generator import load_admm, END_ALIGN_X, MID_ALIGN_X, GOAL, Y_RANGE_MAX, Y_RANGE_MIN, SAMPLE_NUM_B, SAMPLE_NUM_F, HORIZON_B, DT_GLOBAL, X_INI

class PathAdapter:
	def __init__(self, dt_local: float) -> None:
		x_set, u_set, K_set, A_set, B_set = load_admm()
		self.x_set = x_set
		self.u_set = u_set
		self.A_set = A_set
		self.B_set = B_set
		self.K_set = K_set
		self.t_path = 0
		self.t_robot = 0
		self.sample_num = len(self.x_set)
		self.n = len(x_set[0,0])
		self.path_id = int(SAMPLE_NUM_B*2+SAMPLE_NUM_F/2)
		self.dt_local = dt_local
		self.dt_global = DT_GLOBAL
		self.y_span = Y_RANGE_MAX-Y_RANGE_MIN
		self.x_ini = X_INI
		self.goal = GOAL

	def visual_path_adaptation(self, ax: Axes):
		line_set = []
		for i in range(len(self.x_set)):
			if i == self.path_id:
				line, = ax.plot(self.x_set[i,:self.t_path,0], self.x_set[i,:self.t_path,1])
			else:
				line, = ax.plot(self.x_set[i,:self.t_path,0], self.x_set[i,:self.t_path,1], alpha = 0.1)
			line_set.append(line)
		return line_set


	def nominal_ctrl(self, x_cur:np.ndarray):
		dx = x_cur - self.x_set[self.path_id, self.t_path]
		u_nom = self.u_set[self.path_id, self.t_path] + self.K_set[self.path_id,self.t_path] @ dx
		return u_nom


	def update_tracking_policy(self, x_cur:np.ndarray, x_next:np.ndarray):
		#x_cur: robot current states with x, y locations being the first 2 elements
		#x_next: the predicted next state of the robot given control inputs from the cbf safety filter
		
		dx_neighbor = np.zeros((self.sample_num,self.n))
		dx_set = x_cur-self.x_set[:,self.t_path]
		dx_set[:,2] = np.clip(dx_set[:,2]/4,-1,1)
		self.path_id = np.argmin(np.linalg.norm(dx_set,axis=1))
		if x_cur[0]>END_ALIGN_X: 
			#true if the robot in the trajecty constraction region near the target
			#adopt min-norm tracking policies
			nom_dis = -abs(self.x_set[self.path_id,self.t_path,1]-self.goal[1])
			x_dis = -abs(x_cur[1]-self.goal[1])
		elif (self.path_id < 2*SAMPLE_NUM_B and  x_cur[0]<MID_ALIGN_X and x_cur[1]<Y_RANGE_MAX and x_cur[1]>Y_RANGE_MIN):
			#true if the robot is currently tracking one of the backward trajectories in regions behind its initial position
			#adopt horizontal axis tracking policy
			nom_dis = abs(self.x_set[self.path_id,self.t_path,1]-self.x_set[self.path_id,0,1])
			x_dis = abs(x_cur[1]-self.x_set[self.path_id,0,1])
		else:
			#true if the robot is near forward trajectoris
			#adopt vertial axis tracking policy 
			nom_dis = abs(self.x_set[self.path_id,self.t_path,0]-self.x_set[self.path_id,0,0])-0.1
			x_dis = abs(x_cur[0]-self.x_set[self.path_id,0,0])
		
		corner = (x_cur[1]>Y_RANGE_MAX-self.y_span*0.1 or x_cur[1]<Y_RANGE_MIN+self.y_span*0.1) and x_cur[0]<MID_ALIGN_X and self.path_id<2*SAMPLE_NUM_B and np.linalg.norm(dx_set[self.path_id,:])<0.5
		
		if x_dis>(nom_dis-0.5) or corner:
			self.t_path = min(self.t_path+1,int(self.t_robot/self.dt_global*self.dt_local)+1,HORIZON_B-2)
			
			for k in range(self.sample_num):
				dx_neighbor[k] = x_next- self.x_set[k,self.t_path]
				dx_neighbor[k,2] = min(abs(dx_neighbor[k,2]/4),1)
			self.path_id = np.argmin(np.linalg.norm(dx_neighbor,axis=1))
		else:
			self.t_path = min(self.t_path,int(self.t_robot/self.dt_global*self.dt_local)+1,HORIZON_B-2)
		self.t_robot = self.t_robot +1 
		self.path_id = int(self.path_id)
		return 