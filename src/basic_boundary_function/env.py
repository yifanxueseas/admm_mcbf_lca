from typing import List, Optional

import numpy as np
from matplotlib.axes import Axes # type: ignore
from scipy.spatial.transform import Rotation as R # type: ignore

from .obs_gpdf_vec import GassianProcessDistanceField as GPDF # type: ignore


class Env:
    def __init__(self, margin:float=0.0, rho:float=10, load_env=False, env_name:str=None, radius:float=0.6, num_dyn_circle:int=0) -> None:
        """
        Args:
            margin: The margin for the distance function.
            rho: The rho value for the distance function.
            load_env: Load the environment. Defaults to False.
            env_name: If `load_env` is True, specify the environment name.
            radius: The radius of XXX?. Defaults to 0.6.

        Notes:
            Available `env_name` values:
            - hospital
            - vicon
        """
        self.margin = margin
        self.rho = rho
        self.env_name = env_name
        self.radius = radius
        self.num_dyn_circle = num_dyn_circle

        self.gpdf_set:List[GPDF] = []
        self.gpdf_env:List[GPDF] = []
        self.gpdf_mmp:List[GPDF] = []
        self.offset = []
        self.xc_ref = []

        if load_env:
            if env_name == "hospital":
                gpdf_cbf = GPDF()
                gpdf_onM = GPDF()
                gpdf_seat1 = GPDF()
                gpdf_seat2 = GPDF()
                gpdf_seat3 = GPDF()

                line_start = np.array([[-9.5,4.0],[-9.5,3.5],[-6.3, 3.5],[-9.5,1.4],[-9.5,-0.5],[-9.5,-1],[-9.5,-2.9],[-9.5,-3.0],[-9.5,-3.4],[-17.5,-10],[-18,6.2],[-18,7],[-4,7],[-18,-6.3],[-18,-7]])
                line_end = np.array([[-6.3,4.0],[-9.5,1.7],[-6.3,1.7],[-6.3,1.4],[-6.3,-0.5],[-6.3,-1],[-6.3,-2.9],[-6.3,-3.0],[-6.3,-3.4],[-17.5,10],[0,6.2],[0,7],[-4,-7],[-3,-6.3],[-3,-7]])

                res = 200

                pc_coords = gpdf_onM.create_obs(-15.5, 0, 3, 4.2, isLine=False, line_start=line_start[5], line_end=line_end[5], res=res)
                gpdf_onM.update_gpdf(pc_coords)
                self.gpdf_env.append(gpdf_onM)
                self.gpdf_set.append(gpdf_onM)
                self.offset.append(-0.3)
                self.xc_ref.append([-11, 0.0])

                pc_coords = gpdf_seat1.create_obs(isArc=False, isLine=True, line_start=line_start[4:6], line_end=line_end[4:6], res=res)
                gpdf_seat1.update_gpdf(pc_coords)
                self.gpdf_env.append(gpdf_seat1)
                self.gpdf_set.append(gpdf_seat1)
                self.offset.append(-0.3)
                self.xc_ref.append([-8.1, -0.75])

                pc_coords = gpdf_seat2.create_obs(isArc=False, isLine=True, line_start=line_start[0:4], line_end=line_end[0:4], res=res)
                gpdf_seat2.update_gpdf(pc_coords)
                self.gpdf_env.append(gpdf_seat2)
                self.gpdf_set.append(gpdf_seat2)
                self.offset.append(-0.3)
                self.xc_ref.append([-8.1, 2.7])

                pc_coords = gpdf_seat3.create_obs(isArc=False, isLine=True, line_start=line_start[7:9], line_end=line_end[7:9], res=res)
                gpdf_seat3.update_gpdf(pc_coords)
                self.gpdf_env.append(gpdf_seat3)
                self.gpdf_set.append(gpdf_seat3)
                self.offset.append(-0.3)
                self.xc_ref.append([-8.1, -3.25])

                pc_coords = gpdf_cbf.create_obs(isArc=False, isLine=True, line_start=line_start[9:-1], line_end=line_end[9:-1], res=res)
                gpdf_cbf.update_gpdf(pc_coords)
                self.gpdf_env.append(gpdf_cbf)
                self.gpdf_set.append(gpdf_cbf)
                self.offset.append(-0.3)
                self.xc_ref.append([-100., 0.0])
            
            elif env_name == "vicon":
                gpdf_wall = GPDF()
                gpdf_cshape = GPDF()

                # vicon_w_start = np.array([[-2.5,-2.5],[-2.5,2.5],[-2.5,-2.3],[2.9,-2.3]])
                # vicon_w_end = np.array([[2.7,-2.5],[2.7,2.5],[-2.5,2.3],[2.9,2.3]])
                vicon_w_start = np.array([[-5,-3],[-5,3]])
                vicon_w_end = np.array([[5,-3],[5,3]])

                res = 200

                pc_coords = gpdf_cshape.create_arc(0, 0, 1.4, 200, start=-0.73, end=0.73).T
                gpdf_cshape.update_gpdf(pc_coords)
                self.gpdf_set.append(gpdf_cshape) # Non-convex obstacle
                self.gpdf_env.append(gpdf_cshape) # Non-convex obstacle
                self.offset.append(-0.2)
                self.xc_ref.append([0.0, 0.0])
                
                pc_coords = gpdf_wall.create_obs(isArc=False, isLine=True, line_start=vicon_w_start, line_end=vicon_w_end, res=res)
                gpdf_wall.update_gpdf(pc_coords)
                self.gpdf_env.append(gpdf_wall)
                self.gpdf_set.append(gpdf_wall)
                self.offset.append(-0.3)
                self.xc_ref.append([-10, 0.5]) # reference points for specific modulation

            elif env_name == "vicon_fork":
                gpdf_wall = GPDF()

                vicon_w_start = np.array([[-10.0, -3], [-10, 3], [2.0, 3]])
                vicon_w_end   = np.array([[ 10.0, -3], [ 0.0, 3], [10.0, 3]])

                res = 200
                
                pc_coords = gpdf_wall.create_obs(isArc=False, isLine=True, line_start=vicon_w_start, line_end=vicon_w_end, res=res)
                gpdf_wall.update_gpdf(pc_coords)
                self.gpdf_env.append(gpdf_wall)
                self.gpdf_set.append(gpdf_wall)
                self.offset.append(-0.3)
                self.xc_ref.append([-10, 0.5]) # reference points for specific modulation

            else:
                raise ValueError(f"Unknown environment name: {env_name}")
            
        self.update_mmp_gpdf([])
            
    @property
    def mmp(self) -> bool:
        return len(self.gpdf_mmp) > 0
        
    @property
    def num_dyn_mmp(self) -> int:
        if self.mmp:
            return len(self.gpdf_mmp)
        else:
            return 0
    
    def update_mmp_gpdf(self, coords_list: List[np.ndarray]):
        """Update the GPDF list for dynamic obstacles (MMP).

        Args:
            coords_list: The list of coordinates, each element is for one GPDF.
        """
        self.gpdf_mmp = [GPDF(coords) for coords in coords_list]
        self.gpdf_set = self.gpdf_mmp + self.gpdf_env

        self.xc = 100*np.ones((self.num_dyn_mmp,2))
        self.dxc = 0*np.ones((self.num_dyn_mmp,2))

    def h_grad_set(self, x, t=0):
        #x -> Nxdim
        # breakpoint()
        gpdf_num = len(self.gpdf_set)
        for k in range(gpdf_num):
            dis, grad  = self.gpdf_set[k].dis_normal_func(x)
            dis = np.array(dis.reshape(1, -1))
            if k==0:
                dis_set = dis
                grad_set = grad.reshape(1, 2)
            else:
                dis_set = np.concatenate((dis_set, dis),axis=0)
                grad_set = np.concatenate((grad_set, grad.reshape(1, 2)), axis=0)
        if self.env_name in ["vicon", "vicon_fork", "hospital"]:
            for n in range(len(self.xc_ref)):
                dis_set[-(n+1)] = dis_set[-(n+1)] + self.offset[-(n+1)]

        if self.mmp:
            gradt = np.array(self.h_grad_t(grad_set,self.dxc)).reshape(-1,1)
            return dis_set-1, grad_set, gradt, np.zeros(grad_set.shape), np.zeros(grad_set.shape)
        else:
            dis, grad = self.h_gradc(x, t=t)

            grad = np.concatenate((grad, grad_set),axis=0)
            dis = np.concatenate((dis, dis_set), axis=0)
            xc = np.concatenate((self.xc, self.xc_ref),axis=0)
            dxc = np.concatenate((self.dxc, np.zeros(np.array(self.xc_ref).shape)),axis=0)
            gradt = np.array(self.h_grad_t(grad,dxc)).reshape(-1,1)
            return dis-1, grad, gradt, xc, dxc


    def h_grad_vector(self, x, obstacle_idx=-1, dynamic_obstacle=False):
        """Get the gradient of the distance function

        Args:
            x: The current state.
            obstacle_idx: The index of the obstacle to be used, -1 for all obstacles.
            dynamic_obstacle: If True, use dynamic obstacles. Otherwise, use static obstacles.

        Returns:
            The distance and the gradient of the distance function.
        """
        #x -> nxd
        #return -> distance, gradient to x, gradient to t, is closest concave
        if dynamic_obstacle:
            gpdf = self.gpdf_mmp
        else:
            gpdf = self.gpdf_env

        if obstacle_idx < 0: # all obstacles
            dis_set = np.zeros((len(gpdf),len(x)))
            grad_set = np.zeros((len(gpdf),len(x),2))
            for k in range(len(gpdf)):
                dis, grad  = gpdf[k].dis_normal_func(x)
                dis = dis.reshape(1, -1)
                if k==0:
                    dis_set = np.array(dis)
                    grad_set = grad.T[None,:,:]
                else:
                    dis_set = np.concatenate((dis_set,dis),axis=0)
                    grad_set = np.concatenate((grad_set, grad.T[None, :,:]), axis=0)
            if self.env_name in ["vicon", "vicon_fork","hospital"] and not dynamic_obstacle:
                for n in range(len(self.xc_ref)):
                    dis_set[-(n+1)] = dis_set[-(n+1)] + self.offset[-(n+1)]
        else:
            dis_set, grad_set  = gpdf[obstacle_idx].dis_normal_func(x)
            if self.mmp:
                return np.array(dis_set)-1, np.array(grad_set.T)
            dis_set = dis_set.reshape(1,-1)
            grad_set = grad_set.T[None,:,:]

        grad_num = np.sum(np.exp(-self.rho*dis_set.reshape(dis_set.shape[0],dis_set.shape[1],1))*grad_set, axis=0)
        grad_den = np.sum(np.exp(-self.rho*dis_set),axis=0)
        dis_uni = -1/self.rho*np.log(grad_den)-1
        grad_uni = grad_num/grad_den.reshape(-1,1)
        return dis_uni, grad_uni

    def h_gradc(self, x, t=0):
        #x -> 1xdim
        #xc -> Nxdim
        #return -> Nxdim
        if x.shape[0] != 1:
            diff = (-self.xc[:self.num_dyn_circle, np.newaxis] + x).reshape(-1, self.xc[:self.num_dyn_circle].shape[1])
            den = np.linalg.norm(diff, axis=1)
            dis = den.reshape(self.xc[:self.num_dyn_circle].shape[0],x.shape[0])-self.radius-self.margin+1
            grad = (diff/den.reshape(-1,1)).reshape(self.xc[:self.num_dyn_circle].shape[0],x.shape[0],2)
            return dis, grad

        den = np.linalg.norm(x-(self.xc[:self.num_dyn_circle]+self.dxc[:self.num_dyn_circle]*t),axis=1).reshape(self.xc[:self.num_dyn_circle].shape[0],1)
        dis = den - self.radius - self.margin + 1
        grad = (x-self.xc[:self.num_dyn_circle])/den
        return dis, grad
    
    def h_grad_t(self, grad, total_dxc):
        #grad -> Nxdim or MxNxdim
        #xc -> Mxdim
        #return -> Nx1

        if len(grad.shape)>2:
            return -np.sum(grad @ total_dxc.reshape(total_dxc.shape[0], total_dxc.shape[1],1), axis=2)
        else:
            return -np.sum(grad * total_dxc, axis=1)

    def update_xc(self, xc: np.ndarray, dxc: np.ndarray):
        #x -> Nxdim
        self.xc = xc.reshape(self.num_dyn_circle, 2)
        self.dxc = dxc.reshape(self.num_dyn_circle, 2)

    def get_gpdf_xc(self):
        if self.mmp>0:
            return np.concatenate((self.xc, self.xc_ref), axis=0), np.concatenate((self.dxc, np.zeros((len(self.xc_ref),2))), axis=0)
        else:
            return np.array(self.xc_ref), np.zeros((len(self.xc_ref),2))


    
    def compute_weights(self, dist, distMeas_lowerLimit=1, weightPow=1):
        """Compute weights based on a distance measure (with no upper limit)"""
        critical_points = dist <= distMeas_lowerLimit

        if np.sum(critical_points):  # at least one
            if np.sum(critical_points) == 1:
                w = critical_points * 1.0
                return w
            else:
                # TODO: continuous weighting function
                w = critical_points * 1.0 / np.sum(critical_points)
                return w

        dist = dist - distMeas_lowerLimit
        w = (1 / dist) ** weightPow
        if np.sum(w) == 0:
            return w
        w = w / np.sum(w)  # Normalization

        return w
    
    def compute_rel_vel(self, dis, gradt):
        gradt = np.where(gradt>0, 0, gradt)
        weights = self.compute_weights(dis)
        return np.sum(weights*gradt)

    def hes_c(self, x, t=0):
        hes = np.zeros((self.xc.shape[0],2,2))
        diff = x-(self.xc+self.dxc*t)
        dist = np.linalg.norm(diff,axis=1).reshape(self.xc.shape[0],1)
        dist32 = dist**3
        hes[:,0,0] = (1/dist-diff[:,0,None]**2/dist32).squeeze()
        hes[:,1,1] = (1/dist-diff[:,1,None]**2/dist32).squeeze()
        hes[:,0,1] = (-diff[:,0,None]*diff[:,1,None]/dist32).squeeze()
        hes[:,1,0] = hes[:,0,1]
        return hes
    
    def plot_env_standard(self, ax: Axes, color='k', plot_grad_dir=False, obstacle_idx=-1, dynamic_obstacle=False, map_shape_xy=(100, 100), show_grad=False):
        if self.env_name in ["vicon", "vicon_fork"]:
            _x = np.linspace(-3, 4, map_shape_xy[0])
            _y = np.linspace(-3, 4, map_shape_xy[1])
        elif self.env_name == "hospital":
            _x = np.linspace(-18, -4, map_shape_xy[0])
            _y = np.linspace(-8, 8, map_shape_xy[1])
        else:
            raise ValueError(f"Unknown environment name: {self.env_name}")

        if dynamic_obstacle and (self.num_dyn_circle + self.num_dyn_mmp)==0:
            return None, None
        
        X, Y = np.meshgrid(_x, _y)
        dis_mat = np.zeros(X.shape)
        all_xy_coords = np.column_stack((X.ravel(), Y.ravel()))
        dis_mat, normal = self.h_grad_vector(all_xy_coords, obstacle_idx=obstacle_idx, dynamic_obstacle=dynamic_obstacle)
        if plot_grad_dir:
            ax.quiver(X, Y, normal[:, 0], normal[:, 1], color='k', scale=30)
        dis_mat = dis_mat.reshape(map_shape_xy) - 0.0
        if show_grad:
            ctr = ax.contour(X, Y, dis_mat, levels=20, linewidths=1.5)
            ctrf = ax.contourf(X, Y, dis_mat, levels=20, extend='min', alpha=.3)
            ax.clabel(ctr, inline=True)
        else:
            ctr = ax.contour(X, Y, dis_mat, [0], colors=color, linewidths=1.5)
            ctrf = ax.contourf(X, Y, dis_mat, [0, 0.1], colors=['orange','white'], extend='min', alpha=.3)
        return ctr, ctrf
    