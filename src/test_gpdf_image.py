
import numpy as np
import os
from basic_boundary_function.obs_gpdf_vec import GassianProcessDistanceField as GPDF # type: ignore
from basic_boundary_function.obs_gpdf_vec import load_image_as_target
import matplotlib.pyplot as plt # type: ignore

np.random.seed(10)
map_name = '../doc/prob_map'
current_directory = os.getcwd()
folder_path = os.path.join(current_directory)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
save_vid = False


gpdf = GPDF()
target_map = load_image_as_target(map_name + '.png')
pc_map = np.copy(target_map)
pc_map_y, pc_map_x = np.where(pc_map < 254)
pc_coords = np.column_stack((pc_map_x,pc_map_y)) / (len(target_map)/10)

gpdf.update_gpdf(pc_coords)

fig, ax = plt.subplots(1, 3, figsize=(15,5))

target_map_shape = target_map.shape
x = np.linspace(0, 10, target_map_shape[0])
y = np.linspace(0, 10, target_map_shape[1])
X, Y = np.meshgrid(x, y)
all_xy_coords = np.column_stack((X.ravel(), Y.ravel()))
dis_mat, n_vec_mat = gpdf.dis_normal_func(all_xy_coords)

ax[0].pcolormesh(X,Y, target_map, shading='auto', cmap='viridis')
ax[0].set_aspect('equal', 'box')
ax[0].set_xlim([0,10])
ax[0].set_ylim([0,10])

ax[1].pcolormesh(X,Y, dis_mat.reshape(target_map_shape), shading='auto', cmap='jet')
ax[1].set_aspect('equal', 'box')
ax[1].set_xlim([0,10])
ax[1].set_ylim([0,10])

U = n_vec_mat[0]
V = n_vec_mat[1]
U = np.where(dis_mat.flatten()>1.0, U, 0)
V = np.where(dis_mat.flatten()>1.0, V, 0)
U = U.reshape(target_map_shape[0], target_map_shape[1])
V = V.reshape(target_map_shape[0], target_map_shape[1])

ax[2].streamplot(X, Y, U, V, density=2.0)
ax[2].contour(X,Y, dis_mat.reshape(target_map_shape),[1],colors ='k',linewidths=1.0)
ax[2].set_aspect('equal', 'box')
ax[2].set_xlim([0,10])
ax[2].set_ylim([0,10])

plt.show()