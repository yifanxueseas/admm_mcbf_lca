import os
import math
import pathlib
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt # type: ignore

from pkg_moving_object.moving_object import RobotObject, HumanObject
from pkg_moving_object.human_trajs import x_c, dx_c
from configs import CircularRobotSpecification, EnvContrConfiguration, GPDFConfiguration, PedestrianSpecification

from control.cbf_control import OnManCBFController

from basic_boundary_function.env import Env
from basic_boundary_function.onMan_approximation import OnMan_Approx
from path_adapter import PathAdapter

from visualizer.main_plot import PlotInLoop
from visualizer.object import CircularObjectVisualizer



TIMEOUT = 1200

# replace yaml file name for different scenarios
yaml_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', 'env_controller.yaml')
econf = EnvContrConfiguration.from_yaml(yaml_path)

yaml_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', 'gpdf.yaml')
gconf = GPDFConfiguration.from_yaml(yaml_path)
spec_robot_fpath = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', 'spec_robot.yaml')
spec_robot = CircularRobotSpecification.from_yaml(spec_robot_fpath)
spec_human_fpath = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', 'spec_human.yaml')
spec_human = PedestrianSpecification.from_yaml(spec_human_fpath)
path_adapter = PathAdapter(dt_local=spec_robot.ts)

SCENARIO = econf.env_name
ROBOT_TARGET = path_adapter.goal
if econf.dynamics == "differential":
    ROBOT_START = np.array(econf.init_state)
else:
    raise RuntimeError('Undefined robot dynamics.')

AUTORUN = False
SHOW_GRAD = False
MAP_ONLY = True
VB = True
CHECK_MODE = False
DEFAULT_NOMINAL = False    #whether to use user-defined nominal trajectories or default linear dynamical system with no feasiblity guarantuees


env = Env(load_env=True, env_name=SCENARIO, rho=gconf.rho, radius=econf.radius, num_dyn_circle=econf.num_dyn_circle)
om = OnMan_Approx(env, hold_time=econf.hold_time, w=econf.w)

controller = OnManCBFController(threeD_controller=econf.threeD_controller, autotune=econf.autotune, dynamics=econf.dynamics,default_nominal=DEFAULT_NOMINAL)
controller.set_params(nominal_speed=econf.nominal_speed, 
                      sampling_time=spec_robot.ts, 
                      base_margin=econf.base_margin, 
                      target_range=econf.target_range,
                      ve = econf.ve,
                      beta_coef=econf.beta_coef,
                      om_expand_threshold=econf.om_expand_threshold,
                      om_range=econf.om_range,
                      target=ROBOT_TARGET,
                      MCBF=econf.MCBF,
                      dir_num=econf.dir_num)
# controller.set_dynamic_margin(margin_levels=[0.3, 0.1, 0.0])
controller.set_init_data(ROBOT_START, max_iter=TIMEOUT)

if econf.num_dyn_circle:
    HUMAN_STARTS = x_c(59, steps=200)
    om.env.update_xc(x_c(59, steps=200)[:,:2],dx_c(59,steps=200)[:,:2])
else:
    HUMAN_STARTS = []
robot = RobotObject.from_yaml(ROBOT_START, spec_robot_fpath, dynamics=econf.dynamics) # we only need one robot for now
robot_vis = CircularObjectVisualizer(spec_robot.vehicle_width/2, indicate_angle=True if econf.dynamics=="differential" else False)
humans = [HumanObject.from_yaml(hs, spec_human_fpath) for hs in HUMAN_STARTS] # we can have multiple humans
if om.env.num_dyn_circle:
    humans_vis = [CircularObjectVisualizer(econf.radius, indicate_angle=True) for _ in humans]
else:
    humans_vis = [CircularObjectVisualizer(spec_human.human_width/2, indicate_angle=True) for _ in humans]

# visualizer = ... # map and robot should have different sub-visulizers
main_plotter = PlotInLoop(sampling_time=spec_robot.ts, map_only=MAP_ONLY)#, save_to_path='./output.avi', save_params={'skip_frame': int(0.1/DT)-1})
main_plotter.set_env_map(env.plot_env_standard, color='k', plot_grad_dir=False, show_grad=False)
main_plotter.add_object(0, None, (ROBOT_START[0], ROBOT_START[1]), (ROBOT_TARGET[0], ROBOT_TARGET[1]), color='g')


if CHECK_MODE:
    fig_3d = plt.figure()
    ax_3d = plt.axes(projection='3d')

if not MAP_ONLY:
    main_plotter.set_extra_panel(ylabel='Margin')

robot_vis.plot(main_plotter.map_ax, *robot.state, object_color='g')
for human, human_vis in zip(humans, humans_vis):
    human_vis.plot(main_plotter.map_ax, *human.state, object_color='orange')

for kt in range(TIMEOUT):
    if not VB:
        print(f"\r Time step: {kt}/{TIMEOUT}", end='    ')
    else:
        print('='*30)

    if controller.terminal_condition():
        break

    # get the control signal from the controller
    controller.set_state(robot.state)
    start_time = timer()
    u_nom = path_adapter.nominal_ctrl(robot.state)
    run_step_output = controller.run_step(kt, om, u_nom = u_nom, check_mode=CHECK_MODE, vb=VB)
    path_adapter.update_tracking_policy(robot.state, controller.predicted_state)
    u_mod, controller_status = run_step_output[:2]
    if CHECK_MODE:
        _, _, p_set_min, grad_set_min, pi_list, xi_list = run_step_output
        print('p_set_min:', p_set_min)
        # print('pi_list', pi_list)
    if VB:
        print(f"Controller solve time: {timer()-start_time} s")

    # apply the control signal to the robot
    robot.one_step(u_mod)
    if controller_status['isInfeasible']:
        robot_color = 'b'
    elif not controller_status['isSafe']:
        robot_color = 'r'
    else:
        robot_color = '#135e08' # basically green
    robot_vis.update(*robot.state, color=robot_color)
    # update the motion of the humans
    if om.env.num_dyn_circle:
        actions = dx_c(60+kt,steps=200)[:,:2]
    else:
        actions= None
    for i, human, human_vis in zip(range(om.env.num_dyn_circle),humans, humans_vis):
        human.run_step(social_force=None, action=actions[i])
        human_vis.update(*human.state, color=None)
    if om.env.num_dyn_circle:
        om.env.update_xc(x_c(60+kt, steps=200)[:,:2],dx_c(60+kt,steps=200)[:,:2])

    # update the visualizer
    if CHECK_MODE:
        ax_3d.clear()
        if (pi_list is not None) and (xi_list is not None):
            ax_3d.plot(0,0,0,marker='X',markersize=25, markerfacecolor='g', markeredgecolor='g')
            for n in range(len(pi_list)):
                if pi_list[n] > 9000:
                    continue
                if n == np.argmin(pi_list):
                    color = 'r'
                else:
                    color = 'k'
                if len(robot.state)==2:
                    ax_3d.scatter(xi_list[n,:,0,0],xi_list[n,:,0,1],c=color,linewidth=0.2)
                    ax_3d.plot(xi_list[n,-1,0,0],xi_list[n,-1,0,1],marker='X',markersize=10, markerfacecolor=color, markeredgecolor=color)
                else:
                    ax_3d.scatter3D(xi_list[n,:,0,0],xi_list[n,:,0,1],xi_list[n,:,0,2],c=color,linewidth=0.2)
                    ax_3d.plot(xi_list[n,-1,0,0],xi_list[n,-1,0,1],xi_list[n,-1,0,2],marker='o',markersize=10, markerfacecolor=color, markeredgecolor=color)
            env.plot_env_standard(ax_3d,'k')
            env.plot_env_standard(ax_3d, dynamic_obstacle=True, show_grad=SHOW_GRAD)
            ax_3d.set_box_aspect([1,1,1])
            plt.pause(0.01)

    ctr, ctrf = env.plot_env_standard(main_plotter.map_ax, dynamic_obstacle=True, show_grad=SHOW_GRAD, plot_grad_dir=False)
    admm_path_set = path_adapter.visual_path_adaptation(main_plotter.map_ax)

    if False:
    ### activate to display on_manifold auto-parameterization process
        if controller.boundary_points is not None:
            boundary_viz = main_plotter.map_ax.plot(*zip(*controller.boundary_points), 'b-')[0]
        else:
            boundary_viz = None
    else:
        boundary_viz = None

    try:
        e_vec = controller.debug_info['e_vec'][0]

        if e_vec:
            assert isinstance(e_vec, list) 
            e_vec_viz = main_plotter.map_ax.quiver(*robot.state[:2], e_vec[0], e_vec[1], color='r', scale=1, scale_units='xy', angles='xy')
        else:
            e_vec_viz = None
    except:
        e_vec_viz = None
    main_plotter.update_object(0, kt, u_mod, robot.state, controller.debug_info['current_margin'], None, None)
    main_plotter.plot_in_loop(
        polygonal_dyn_obstacle_list=None,
        other_plt_objects=[ctr, ctrf, boundary_viz, e_vec_viz]+admm_path_set,
        time=kt*spec_robot.ts, autorun=AUTORUN, zoom_in=None
    )


print()

input("Press Enter to continue...")

if CHECK_MODE:
    fig_3d.show()
    plt.close(fig_3d)

if main_plotter.is_active:
    main_plotter.show()
    main_plotter.close()

print("Done!")

    
