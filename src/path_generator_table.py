import numpy as np
import pandas as pd
from admm.admm_lca import admm_lca, run_parallel_admm
from admm.model_defn import *
from trajax.integrators import rk4
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

path_type = 'f'							#auto is only suitable for 'f', forward trajectories
DT_GLOBAL = 0.5							#discrete timestep used by the discrete-time global path planner
horizon = 16							#horizon of the forward trajectories
u_max = np.array([1,2])					#input minimums of the robot enforced by the global path planner
u_min = np.array([0,-2])				#input maximums of the robot enforced by the global path planner
sample_num = 5							#the number of trajectories that spans regions on the back of the robot
x_ini_num = 5

# Define variables for the problem specification
dynamics = rk4(unicycle, dt=DT_GLOBAL)	#robot dynamics input to the global path planner
X_INI_POS = [-1.6, 0.]					#robot initial states
X_INI_LIST = []
GOAL = np.array([2.0, 0.0])				#goal location in (x,y)
for i in range(x_ini_num):
	theta = i*2*np.pi/x_ini_num
	Rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
	NEW_POS = GOAL + Rot@(X_INI_POS-GOAL)
	X_INI_LIST.append([NEW_POS[0], NEW_POS[1], i*2*np.pi/x_ini_num])
u0 = np.zeros(2)						#robot input initialization
rho = 50
m = 2									#robot input dimension
n = 3									#robot state dimension
runtime_list = []

print(X_INI_LIST)

def load_admm():
	gain_K_set = []
	x_cur_set = []
	u_cur_set = []
	A_set = []
	B_set = []
	for j in range(len(X_INI_LIST)):
		for k in range(sample_num):
			df = pd.read_csv("../data/offline_admm"+path_type+str(j)+"_table.csv")
			x = np.repeat(np.array(X_INI_LIST[j]).reshape(1,n), repeats=horizon, axis=0)
			u = np.zeros((horizon,m))
			A = np.zeros((horizon, n, n))
			B = np.zeros((horizon,n,m))
			gain_K = np.zeros((horizon,m,n))

			for l in range(n):
				x[-horizon:,l] = df['x'+str(k)+str(l)]
			for l in range(m):
				u[-horizon:,l] = df['u'+str(k)+str(l)]
			for l1 in range(n):
				for l2 in range(n):
					A[-horizon:,l1,l2] = df['A'+str(k)+str(l1)+str(l2)]
				for l2 in range(m):
					B[-horizon:,l1,l2] = df['B'+str(k)+str(l1)+str(l2)]
			for l1 in range(m):
				for l2 in range(n):
					gain_K[-horizon:, l1, l2] = df['gain'+str(k)+str(l1)+str(l2)]
				
				
			gain_K_set.append(gain_K)
			x_cur_set.append(x)
			u_cur_set.append(u)
			A_set.append(A)
			B_set.append(B)

	x_cur_set = np.array(x_cur_set)
	u_cur_set = np.array(u_cur_set)
	A_set = np.array(A_set)
	B_set = np.array(B_set)
	gain_K_set = np.array(gain_K_set)
	return x_cur_set, u_cur_set, gain_K_set, A_set, B_set


def generate_path():
	df = pd.DataFrame()
	
	for j in range(len(X_INI_LIST)):
		admm_obj = admm_lca(dynamics, horizon, DT_GLOBAL, m, n, np.array(X_INI_LIST[j]), u0, GOAL, rho, idx=(0, 1), umax=u_max, umin=u_min)
		# nominal_ctrl = run_parallel_admm(admm_obj, dyn="1st",points=p_list[j],use_points=True,horizon=horizon[j])
		# breakpoint()
		nominal_ctrl = run_parallel_admm(admm_obj, num_samples=sample_num,dispersion_size = 0.78,dyn="1st",num_wayp=5, horizon=horizon)
		for k in range(sample_num):
			gain_K, gain_k, x, u, r, A, B, runtime = nominal_ctrl[k]

			for l in range(n):
				df['x'+str(k)+str(l)]=pd.Series(x[:,l])
			for l in range(m):
				df['u'+str(k)+str(l)]=pd.Series(u[:,l])
			for l1 in range(n):
				for l2 in range(n):
					df['A'+str(k)+str(l1)+str(l2)]=pd.Series(A[:,l1,l2])
				for l2 in range(m):
					df['B'+str(k)+str(l1)+str(l2)]=pd.Series(B[:,l1,l2])
			for l1 in range(m):
				for l2 in range(n):
					df['gain'+str(k)+str(l1)+str(l2)]=pd.Series(gain_K[:,l1,l2])
			runtime_list.append(runtime)

		df.to_csv("../data/offline_admm"+path_type+str(j)+"_table.csv")


if __name__ == '__main__':
	generate_path()
	x_cur_set, u_cur_set, gain_K_set, A_set, B_set = load_admm()
	repeat = 0
	print(len(x_cur_set))
	print(runtime_list)
	print(len(runtime_list))
	print("runtime avg", np.mean(runtime_list))
	for i in range(len(x_cur_set)):
		plt.plot(x_cur_set[i,:,0], x_cur_set[i,:,1])

	for i in range(x_ini_num):
		plt.plot([X_INI_LIST[i][0], X_INI_LIST[i][0]+np.cos(X_INI_LIST[i][2])],[X_INI_LIST[i][1], X_INI_LIST[i][1]+np.sin(X_INI_LIST[i][2])], scalex=30, scaley=30 )

	plt.show()

