import math
from typing import Optional, Callable, TypedDict, Union, Tuple
# from typing import Union, Callable, Optional, Type, Sequence, Tuple, List

import numpy as np

# Vis import
import cv2 # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as patches # type: ignore
from matplotlib.lines import Line2D # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore
from matplotlib.axes import Axes # type: ignore
from matplotlib.patches import Patch # type: ignore


class SaveParams(TypedDict):
    fps: int
    dpi: int
    codec: str
    frame_size: Tuple[int, int]
    skip_frame: int


def figure_formatter(
        window_title: str, 
        num_axes_per_column:Optional[list]=None, 
        num_axes_per_row:Optional[list]=None, 
        dpi:Optional[int]=None,
        figure_size:Optional[Tuple[float, float]]=None):
    """ Generate a figure with a given format.

    Args:
        num_axes_per_column: The length of the list is the number of columns of the figure. 
            E.g. [1,3] means the figure has two columns and with 1 and 3 axes respectively.
        num_axes_per_row: The length of the list is the number of rows of the figure.
            E.g. [1,3] means the figure has two rows and with 1 and 3 axes respectively.
        figure_size: If None, then figure size is adaptive.

    Returns:
        axis_format: List of axes lists,
        - If use `num_axes_per_column`, axes[i][j] means the j-th axis in the i-th column.
        - If use `num_axes_per_row`, axes[i][j] means the j-th axis in the i-th row.
        
    Note:
        `num_axes_per_column` and `num_axes_per_row` cannot be both specified.
    """
    if (num_axes_per_column is None) and (num_axes_per_row is None):
        raise ValueError("Either `num_axes_per_column` or `num_axes_per_row` must be specified.")
    elif (num_axes_per_column is not None) and (num_axes_per_row is not None):
        raise ValueError("Cannot specify both `num_axes_per_column` and `num_axes_per_row`.")
    
    if num_axes_per_column is not None:
        n_col   = len(num_axes_per_column)
        n_row   = np.lcm.reduce(num_axes_per_column) # least common multiple
        row_res = [int(n_row//x) for x in num_axes_per_column] # greatest common divider
    elif num_axes_per_row is not None:
        n_row   = len(num_axes_per_row)
        n_col   = np.lcm.reduce(num_axes_per_row)
        col_res = [int(n_col//x) for x in num_axes_per_row]

    fig = plt.figure(figsize=figure_size, constrained_layout=True, dpi=dpi)
    assert fig.canvas.manager is not None
    fig.canvas.manager.set_window_title(window_title)
    gs = GridSpec(n_row, n_col, figure=fig)

    axis_format:list[list] = []
    if num_axes_per_column is not None:
        for i in range(n_col):
            axis_format.append([])
            for j in range(num_axes_per_column[i]):
                row_start = j    *row_res[i]
                row_end   = (j+1)*row_res[i]
                axis_format[i].append(fig.add_subplot(gs[row_start:row_end, i]))
    elif num_axes_per_row is not None:
        for i in range(n_row):
            axis_format.append([])
            for j in range(num_axes_per_row[i]):
                col_start = j    *col_res[i]
                col_end   = (j+1)*col_res[i]
                axis_format[i].append(fig.add_subplot(gs[i, col_start:col_end]))
    return fig, gs, axis_format


class PlotInLoop:
    def __init__(self, sampling_time: float, map_only=False, save_to_path:Optional[str]=None, save_params:Optional[Union[SaveParams, dict]]=None) -> None:
        """
        Args:
            sampling_time: The time interval between each time step.
            map_only: If True, only the map will be plotted.
            save_to_path: If not None, the plot will be saved to the path as a video.
            save_params: The parameters for saving the plot as a video, such as `fps`, `dpi`, `codec`, `frame_size`, `skip_frame`.

        Attributes:
            plot_dict_pre   : A dictionary of all plot objects which need to be manually flushed.
            plot_dict_temp  : A dictionary of all plot objects which only exist for one time step.
            plot_dict_inloop: A dictionary of all plot objects which update (append) every time step.

        Note:
            Figure layout:
            ```
                |================|================|
                |     Speed      |                |
                |================|                |
                |  Angular speed |       Map      |
                |================|                |
                |     Extra      |                |
                |================|================|
            ```

        TODO:
            - Methods to flush part of the plot and to destroy an object in case it is not active.
        """
        self.ts = sampling_time
        self.map_only = map_only
        self.init_video_writer(save_to_path, save_params)

        if save_to_path is None:
            dpi = None
        else:
            dpi = self.save_params['dpi']

        if map_only:
            self.fig, self.map_ax = plt.subplots(dpi=dpi)
        else:
            self.fig, self.gs, axis_format = figure_formatter('PlotInLoop', [3,1], dpi=dpi) # type: ignore

            self.vel_ax:Axes = axis_format[0][0]
            self.omega_ax:Axes = axis_format[0][1]
            self.extra_ax:Axes = axis_format[0][2]
            self.map_ax:Axes = axis_format[1][0] # type: ignore

            [ax.grid(visible=True) for ax in [self.vel_ax, self.omega_ax, self.extra_ax]] # type: ignore
            [ax.set_xlabel('Time [s]') for ax in [self.vel_ax, self.omega_ax, self.extra_ax]]
            self.vel_ax.set_ylabel('Velocity [m/s]')
            self.omega_ax.set_ylabel('Angular velocity [rad/s]')
            self.extra_ax.set_ylabel('[Not defined]')

        self.map_ax.set_xlabel('X [m]', fontsize=15)
        self.map_ax.set_ylabel('Y [m]', fontsize=15)
        self.map_ax.axis('equal')
        self.map_ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        self.map_ax.tick_params(axis='y', which='both', left=True, labelleft=True)

        self.remove_later:list[Patch] = [] # patches need to be flushed
        self.plot_dict_pre:dict = {}    # flush for every life cycle
        self.plot_dict_temp:dict = {}   # flush for every time step
        self.plot_dict_inloop:dict = {} # update every time step, flush for every life cycle
        self.mask_img = None
        self.skip_counter = 0

    @property
    def is_active(self):
        return plt.fignum_exists(self.fig.number)
    
    def show(self):
        if self.save_to_path is None:
            self.fig.show()

    def close(self):
        if self.save_to_path is not None:
            self.video_writer.release()
        plt.close(self.fig)

    def init_video_writer(self, save_to_path: Optional[str], save_params: Optional[Union[SaveParams, dict]]):
        """Initialize the video writer if the path is not None.

        Attributes:
            save_params: The parameters for saving the plot as a video, such as `fps`, `dpi`, `codec`, `frame_size`, `skip_frame`.
            video_writer: The video writer object.
        """
        default_frame_size = (1280, 960)
        default_fps = 5
        default_dpi = 200
        default_codec = 'MJPG'
        default_skip_frame = 0 # skip every n frames, 0 means no skip
        self.save_to_path = save_to_path
        if save_to_path is not None:
            if save_params is None:
                self.save_params = SaveParams(
                    fps=default_fps,
                    dpi=default_dpi,
                    codec=default_codec,
                    frame_size=default_frame_size,
                    skip_frame=default_skip_frame
                )
            else:
                self.save_params = SaveParams(
                    fps=save_params.get('fps', default_fps),
                    dpi=save_params.get('dpi', default_dpi),
                    codec=save_params.get('codec', default_codec),
                    frame_size=save_params.get('frame_size', default_frame_size),
                    skip_frame=save_params.get('skip_frame', default_skip_frame)
                )
            fourcc = cv2.VideoWriter_fourcc(*self.save_params['codec'])
            self.video_writer = cv2.VideoWriter(save_to_path, fourcc, self.save_params['fps'], self.save_params['frame_size'])

    def set_extra_panel(self, ylabel: str):
        self.extra_ax.set_ylabel(ylabel)

    def set_env_map(self, plot_function: Callable, *args, **kwargs):
        """Call the external plot function to plot the static map.
        
        The first argument of the function is set to the map axis."""
        plot_function(self.map_ax, *args, **kwargs)

    
    def add_object(self, object_id, ref_traj: Optional[np.ndarray], start: Optional[Tuple], end: Optional[Tuple], color):
        """This function should be called for each (new) object that needs to be plotted.

        Args:
            ref_traj: Each row is a state
            color: Matplotlib style color
        """
        if object_id in list(self.plot_dict_pre):
            raise ValueError(f'[{self.__class__.__name__}] Object ID {object_id} exists!')
        
        ref_line = None
        if ref_traj is not None:
            ref_line,  = self.map_ax.plot(ref_traj[:,0], ref_traj[:,1],   color=color, linestyle='--', label='Ref trajectory')
        start_pt = None
        if start is not None:
            start_pt,  = self.map_ax.plot(start[0], start[1], marker='*', color=color, markersize=15, alpha=0.2,  label='Start')
        end_pt = None
        if end is not None:
            end_pt,    = self.map_ax.plot(end[0],   end[1],   marker='X', color=color, markersize=15, alpha=0.2,  label='End')
        self.plot_dict_pre[object_id] = [ref_line, start_pt, end_pt]

        self.plot_dict_inloop[object_id] = []
        if not self.map_only:
            vel_line,   = self.vel_ax.plot([], [],   marker='o', color=color)
            omega_line, = self.omega_ax.plot([], [], marker='o', color=color)
            extra_line,  = self.extra_ax.plot([], [],  marker='o', color=color)
            self.plot_dict_inloop[object_id].extend([vel_line, omega_line, extra_line])
        past_line,  = self.map_ax.plot([], [],  marker='.', linestyle='None', color=color)
        self.plot_dict_inloop[object_id].append(past_line)

        ref_line_now,  = self.map_ax.plot([], [], marker='x', linestyle='None', color=color)
        pred_line,     = self.map_ax.plot([], [], marker='+', linestyle='None', color=color)
        self.plot_dict_temp[object_id] = [ref_line_now, pred_line]

    def update_object(self, object_id, kt, action, state, extra: Optional[float], pred_states: Optional[np.ndarray], current_ref_traj: Optional[np.ndarray]):
        """Update the object with the new data.

        Args:
            action: velocity and angular velocity
            pred_states: np.ndarray, each row is a state
            current_ref_traj: np.ndarray, each row is a state
        """
        if object_id not in list(self.plot_dict_pre):
            raise ValueError(f'[{self.__class__.__name__}] Object ID {object_id} does not exist!')
        if extra is None:
            extra = 0.0

        if self.map_only:
            update_list = [state]
        else:
            update_list = [action[0], action[1], extra, state]
        for new_data, line in zip(update_list, self.plot_dict_inloop[object_id]):
            assert isinstance(line, Line2D)
            if isinstance(new_data, (int, float)):
                line.set_xdata(np.append(line.get_xdata(),  kt*self.ts))
                line.set_ydata(np.append(line.get_ydata(),  new_data))
            else:
                line.set_xdata(np.append(line.get_xdata(),  new_data[0]))
                line.set_ydata(np.append(line.get_ydata(),  new_data[1]))

        if (current_ref_traj is not None) and (pred_states is not None):
            temp_list = [current_ref_traj, pred_states]
            for new_data, line in zip(temp_list, self.plot_dict_temp[object_id]):
                assert isinstance(line, Line2D)
                line.set_data(new_data[:, 0], new_data[:, 1])


    def plot_in_loop(self, 
                     mask:Optional[np.ndarray]=None,
                     mask_extent:Optional[list]=None,
                     polygonal_dyn_obstacle_list=None, 
                     elliptical_dyn_obstacle_list=None, 
                     other_plt_objects:Optional[list]=None,
                     time:Optional[float]=None, autorun=False, zoom_in=None):
        """
        Args:
            polygonal_dyn_obstacle_list: List of polygonal obstacles.
            elliptical_dyn_obstacle_list: List of obstacle_list, where each one has N_hor predictions.
            time: current time (actual time).
            autorun: if true, the plot will not pause.
            zoom_in: if not None, the map will be zoomed in [xmin, xmax, ymin, ymax].
        """
        if time is not None:
            self.map_ax.set_title(f'Time: {time:.2f}s / {time/self.ts:.2f}')

        if zoom_in is not None:
            self.map_ax.set_xlim(zoom_in[0:2])
            self.map_ax.set_ylim(zoom_in[2:4])

        if mask is not None:
            if self.mask_img is None:
                if mask_extent is not None:
                    self.mask_img = self.map_ax.imshow(mask, cmap='gray', alpha=0.5, extent=mask_extent)
                else:
                    self.mask_img = self.map_ax.imshow(mask, cmap='gray', alpha=0.5)
            else:
                self.mask_img.set_data(mask)
        elif self.mask_img is not None:
            self.mask_img.remove()

        if other_plt_objects is not None:
            self.remove_later.extend([x for x in other_plt_objects if x is not None])

        if polygonal_dyn_obstacle_list is not None:
            for obstacle in polygonal_dyn_obstacle_list:
                this_polygon = patches.Polygon(obstacle, color='r', alpha=0.5, label='Obstacle')
                # raw_poly = self.map_ax.plot(obstacle[:,0], obstacle[:,1], 'r.')
                self.map_ax.add_patch(this_polygon)
                self.remove_later.append(this_polygon)
                # self.remove_later.append(raw_poly[0])

        if elliptical_dyn_obstacle_list is not None:
            for obstacle_list in elliptical_dyn_obstacle_list: # each "obstacle_list" has N_hor predictions
                current_one = True
                for al, pred in enumerate(obstacle_list):
                    x,y,rx,ry,angle,alpha = pred
                    if current_one:
                        this_color = 'k'
                    else:
                        this_color = 'r'
                    if alpha > 0:
                        pos = (x,y)
                        this_ellipse = patches.Ellipse(pos, rx*2, ry*2, angle=angle/(2*math.pi)*360, color=this_color, alpha=max(8-al,1)/20, label='Obstacle')
                        self.map_ax.add_patch(this_ellipse)
                        self.remove_later.append(this_ellipse)
                    current_one = False

        ### Autoscale
        if not self.map_only:
            for ax in [self.vel_ax, self.omega_ax, self.extra_ax]:
                x_min = min(ax.get_lines()[0].get_xdata())
                x_max = max(ax.get_lines()[0].get_xdata())
                y_min = min(ax.get_lines()[0].get_ydata())
                y_max = max(ax.get_lines()[0].get_ydata())
                for line in ax.get_lines():
                    if x_min  > min(line.get_xdata()):
                        x_min = min(line.get_xdata())
                    if x_max  < max(line.get_xdata()):
                        x_max = max(line.get_xdata())
                    if y_min  > min(line.get_ydata()):
                        y_min = min(line.get_ydata())
                    if y_max  < max(line.get_ydata()):
                        y_max = max(line.get_ydata())
                ax.set_xlim([x_min, x_max+1e-3])
                ax.set_ylim([y_min, y_max+1e-3])

        self.fig.canvas.draw()
        if self.save_to_path is None:
            plt.pause(0.01)
            if not autorun:
                while not plt.waitforbuttonpress():
                    pass

        if self.save_to_path is not None:
            if self.skip_counter >= self.save_params['skip_frame']:
                self.skip_counter = 0
                save_img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                save_img = save_img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                save_img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                self.video_writer.write(cv2.resize(save_img_bgr, self.save_params['frame_size']))
            else:
                self.skip_counter += 1

        for j in range(len(self.remove_later)): # robot and dynamic obstacles (predictions)
            self.remove_later[j].remove()
        self.remove_later = []

