import numpy as np
import pandas as pd
from admm.admm_lca import admm_lca, run_parallel_admm
from admm.model_defn import *
from trajax.integrators import rk4
import matplotlib.pyplot as plt


path_type = 'f'							#'b' for backwards trajectory, 'f' for forward trajectories, 'a' for both
DT_GLOBAL = 0.5							#discrete timestep used by the discrete-time global path planner
HORIZON_B = 72							#horizon of the backwards trajectories
HORIZON_F = 40							#horizon of the forward trajectories
u_max = np.array([1,1])					#input minimums of the robot enforced by the global path planner
u_min = np.array([0,-1])				#input maximums of the robot enforced by the global path planner
SAMPLE_NUM_B = 4						#the number of trajectories that spans regions on the back of the robot
SAMPLE_NUM_F = 15						#the number of trajectories that spans regions on the back of the robot

# Define variables for the problem specification
dynamics = rk4(unicycle, dt=DT_GLOBAL)	#robot dynamics input to the global path planner
X_INI = [-14., 0., 0.]					#robot initial state
GOAL = np.array([-5., 0.])				#goal location in (x,y)
u0 = np.zeros(2)						#robot input initialization
rho = 50
m = 2									#robot input dimension
n = 3									#robot state dimension

horizon = []
p_list = []
sample_num = []

#selected waypoint sets: 
MID_ALIGN_X = -13.0
END_ALIGN_X = -7.0
Y_RANGE_MAX = 5.6
Y_RANGE_MIN = -5.6
wayp_num_b = 9
pb_l_list = np.zeros((SAMPLE_NUM_B, wayp_num_b,2))
pb_l_list[0] = np.array([[-14.0, 0.5],[-14.0, 1.5],[-14.0, 3.0],[-14.0, 4.5],[-14.0, 5.2],[MID_ALIGN_X,Y_RANGE_MAX],[-10.0,Y_RANGE_MAX],[END_ALIGN_X,Y_RANGE_MAX],[-6.0,2*Y_RANGE_MAX/3]])
pb_l_list[1] = np.array([[-14.5, 0.5],[-15.0, 1.5],[-15.0, 3.0],[-15.0, 4.5],[-14.5, 5.2],[MID_ALIGN_X,Y_RANGE_MAX],[-10.0,Y_RANGE_MAX],[END_ALIGN_X,Y_RANGE_MAX],[-6.0,2*Y_RANGE_MAX/3]])
pb_l_list[2] = np.array([[-15.0, 0.5],[-16.0, 1.5],[-16.0, 3.0],[-16.0, 4.5],[-15.0, 5.2],[MID_ALIGN_X,Y_RANGE_MAX],[-10.0,Y_RANGE_MAX],[END_ALIGN_X,Y_RANGE_MAX],[-6.0,2*Y_RANGE_MAX/3]])
pb_l_list[3] = np.array([[-15.5, 0.5],[-17.0, 1.5],[-17.0, 3.0],[-17.0, 4.5],[-16.0, 5.2],[MID_ALIGN_X,Y_RANGE_MAX],[-10.0,Y_RANGE_MAX],[END_ALIGN_X,Y_RANGE_MAX],[-6.0,2*Y_RANGE_MAX/3]])

pb_r_list = np.zeros((SAMPLE_NUM_B, wayp_num_b,2))
pb_r_list[0] = np.array([[-15.5,-0.5],[-17.0,-1.5],[-17.0,-3.0],[-17.0,-4.5],[-16.0,-5.2],[MID_ALIGN_X,Y_RANGE_MIN],[-10.0,Y_RANGE_MIN],[END_ALIGN_X,Y_RANGE_MIN],[-6.0,2*Y_RANGE_MIN/3]])
pb_r_list[1] = np.array([[-15.0,-0.5],[-16.0,-1.5],[-16.0,-3.0],[-16.0,-4.5],[-15.0,-5.2],[MID_ALIGN_X,Y_RANGE_MIN],[-10.0,Y_RANGE_MIN],[END_ALIGN_X,Y_RANGE_MIN],[-6.0,2*Y_RANGE_MIN/3]])
pb_r_list[2] = np.array([[-14.5,-0.5],[-15.0,-1.5],[-15.0,-3.0],[-15.0,-4.5],[-14.5,-5.2],[MID_ALIGN_X,Y_RANGE_MIN],[-10.0,Y_RANGE_MIN],[END_ALIGN_X,Y_RANGE_MIN],[-6.0,2*Y_RANGE_MIN/3]])
pb_r_list[3] = np.array([[-14.0,-0.5],[-14.0,-1.5],[-14.0,-3.0],[-14.0,-4.5],[-14.0,-5.2],[MID_ALIGN_X,Y_RANGE_MIN],[-10.0,Y_RANGE_MIN],[END_ALIGN_X,Y_RANGE_MIN],[-6.0,2*Y_RANGE_MIN/3]])

wayp_num_f = 5
pf_list = np.zeros((SAMPLE_NUM_F,wayp_num_f,2))
pf_list[0] = np.array([[-13.5,-2.8],[MID_ALIGN_X,Y_RANGE_MIN],[-10.0,Y_RANGE_MIN],[END_ALIGN_X,Y_RANGE_MIN],[-6.0,2*Y_RANGE_MIN/3]])
pf_list[1] = np.array([[-13.5,-2.4],[MID_ALIGN_X,-4.8],[-10.0,-4.8],[END_ALIGN_X,-4.8],[-6.0,-3.2]])
pf_list[2] = np.array([[-13.5,-2.0],[MID_ALIGN_X,-4.0],[-10.0,-4.0],[END_ALIGN_X,-4.0],[-6.0,-2.6]])
pf_list[3] = np.array([[-13.5,-1.6],[MID_ALIGN_X,-3.2],[-10.0,-3.2],[END_ALIGN_X,-3.2],[-6.0,-2.1]])
pf_list[4] = np.array([[-13.5,-1.2],[MID_ALIGN_X,-2.4],[-10.0,-2.4],[END_ALIGN_X,-2.4],[-6.0,-1.6]])
pf_list[5] = np.array([[-13.5,-0.8],[MID_ALIGN_X,-1.6],[-10.0,-1.6],[END_ALIGN_X,-1.6],[-6.0,-1.1]])
pf_list[6] = np.array([[-13.5,-0.4],[MID_ALIGN_X,-0.8],[-10.0,-0.8],[END_ALIGN_X,-0.8],[-6.0,-0.6]])
pf_list[7] = np.array([[-13.5, 0.0],[MID_ALIGN_X, 0.0],[-10.0, 0.0],[END_ALIGN_X, 0.0],[-6.0, 0.0]])
pf_list[8] = np.array([[-13.5, 0.4],[MID_ALIGN_X, 0.8],[-10.0, 0.8],[END_ALIGN_X, 0.8],[-6.0, 0.6]])
pf_list[9] = np.array([[-13.5, 0.8],[MID_ALIGN_X, 1.6],[-10.0, 1.6],[END_ALIGN_X, 1.6],[-6.0, 1.1]])
pf_list[10] = np.array([[-13.5,1.2],[MID_ALIGN_X, 2.4],[-10.0, 2.4],[END_ALIGN_X, 2.4],[-6.0, 1.6]])
pf_list[11] = np.array([[-13.5,1.6],[MID_ALIGN_X, 3.2],[-10.0, 3.2],[END_ALIGN_X, 3.2],[-6.0, 2.1]])
pf_list[12] = np.array([[-13.5,2.0],[MID_ALIGN_X, 4.0],[-10.0, 4.0],[END_ALIGN_X, 4.0],[-6.0, 2.6]])
pf_list[13] = np.array([[-13.5,2.4],[MID_ALIGN_X, 4.8],[-10.0, 4.8],[END_ALIGN_X, 4.8],[-6.0, 3.2]])
pf_list[14] = np.array([[-13.5,2.8],[MID_ALIGN_X,Y_RANGE_MAX],[-10.0,Y_RANGE_MAX],[END_ALIGN_X,Y_RANGE_MAX],[-6.0,2*Y_RANGE_MAX/3]])


#problem initialization 
if path_type == 'b':
	repeat_num = 2
	p_list.append(pb_l_list)
	p_list.append(pb_r_list)
	horizon.append(HORIZON_B)
	horizon.append(HORIZON_B)
	sample_num.append(SAMPLE_NUM_B)
	sample_num.append(SAMPLE_NUM_B)
elif path_type == 'b0':
	repeat_num = 1
	p_list.append(pb_l_list)
	horizon.append(HORIZON_B)
	sample_num.append(SAMPLE_NUM_B)
elif path_type == 'b1':
	repeat_num = 1
	p_list.append(pb_r_list)
	horizon.append(HORIZON_B)
	sample_num.append(SAMPLE_NUM_B)
elif path_type == 'f':
	repeat_num = 1
	p_list.append(pf_list)
	horizon.append(HORIZON_F)
	sample_num.append(SAMPLE_NUM_F)
else:
	repeat_num = 3
	p_list.append(pb_l_list)
	p_list.append(pb_r_list)
	p_list.append(pf_list)
	horizon.append(HORIZON_B)
	horizon.append(HORIZON_B)
	horizon.append(HORIZON_F)
	sample_num.append(SAMPLE_NUM_B)
	sample_num.append(SAMPLE_NUM_B)
	sample_num.append(SAMPLE_NUM_F)


def load_admm():
	gain_K_set = []
	x_cur_set = []
	u_cur_set = []
	A_set = []
	B_set = []
	for i in range(repeat_num):
		df = pd.read_csv("../data/offline_admm"+path_type+str(i)+"_hospital.csv")
		for k in range(sample_num[i]):
			x = np.repeat(np.array(X_INI).reshape(1,n), repeats=max(horizon), axis=0)
			u = np.zeros((max(horizon),m))
			A = np.zeros((max(horizon), n, n))
			B = np.zeros((max(horizon),n,m))
			gain_K = np.zeros((max(horizon),m,n))

			for l in range(n):
				x[-horizon[i]:,l] = df['x'+str(k)+str(l)]
			for l in range(m):
				u[-horizon[i]:,l] = df['u'+str(k)+str(l)]
			for l1 in range(n):
				for l2 in range(n):
					A[-horizon[i]:,l1,l2] = df['A'+str(k)+str(l1)+str(l2)]
				for l2 in range(m):
					B[-horizon[i]:,l1,l2] = df['B'+str(k)+str(l1)+str(l2)]
			for l1 in range(m):
				for l2 in range(n):
					gain_K[-horizon[i]:, l1, l2] = df['gain'+str(k)+str(l1)+str(l2)]
			
			
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
	for j in range(repeat_num):
		df = pd.DataFrame()

		admm_obj = admm_lca(dynamics, horizon[j], DT_GLOBAL, m, n, np.array(X_INI), u0, GOAL, rho, idx=(0, 1), umax=u_max, umin=u_min)
		nominal_ctrl = run_parallel_admm(admm_obj, dyn="1st",points=p_list[j],use_points=True,horizon=horizon[j])

		for k in range(sample_num[j]):
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

		df.to_csv("../data/offline_admm"+path_type+str(j)+"_hospital.csv")


if __name__ == '__main__':
	generate_path()
	x_cur_set, u_cur_set, gain_K_set, A_set, B_set = load_admm()
	repeat = 0
	for i in range(len(x_cur_set)):
		plt.plot(x_cur_set[i,:,0], x_cur_set[i,:,1])
	for i in range(len(p_list)):
		for j in range(sample_num[i]):
			plt.scatter(p_list[i][j,:,0], p_list[i][j,:,1])


	plt.show()

