import numpy as np
import matplotlib.pyplot as plt

def generate_lines(x1, y1, x2, y2, num_points):
    """
    Generate points along a line connecting (x1, y1) and (x2, y2).

    Parameters:
    x1, y1: Coordinates of the first point.
    x2, y2: Coordinates of the second point.
    num_points: Number of points to generate along the line.

    Returns:
    x_points, y_points: Arrays of x and y coordinates of the generated points.
    """
    x_points = np.linspace(x1, x2, num_points)
    y_points = np.linspace(y1, y2, num_points)
    return x_points, y_points


def generate_semicircles(width=20, init_state=(0, 0), goal_state=(-10, 0)):
    """
    Given the total width of room and a a pivot, generate curves with varying radii.
    """
    # Parameters
    radii = [width/4, width/5, width/6, width/8]  # Different radii
    num_samples = 2 * len(radii) # Number of samples
    num_wp = 11 # Number of waypoints

    alpha = np.arctan2(goal_state[1] - init_state[1], goal_state[0] - init_state[0]) - np.pi/2  # Angle of the line connecting pivot and goal
    print(f"Angle: {np.degrees(alpha)}")
    # if alpha == np.pi/2 or alpha == -np.pi/2 or alpha == 0 or alpha == np.pi:
    #     pivot = init_state  # Pivot point
    #     goal = goal_state  # Goal point
    # else:
    rot_mat = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])  # Rotation matrix
    inv_rot_mat = np.linalg.inv(rot_mat)  # Inverse rotation matrix
    pivot = rot_mat @ init_state  # Pivot point
    goal = rot_mat @ goal_state  # Goal point

    wp = np.zeros((num_samples, num_wp, 2))  # Waypoints
    
    plt.figure(figsize=(8, 5))
    
    # Colors for differentiation
    colors = ['b', 'g', 'r', 'm']
    
    for i, radius in enumerate(radii):
        angles = np.arange(0, 181, 30)  # Angles from left (0°) to right (180°)

        # Compute right points
        xr_forward = pivot[0] + radius - radius * np.cos(np.radians(angles))
        if alpha == 0.0 or alpha == np.pi or alpha == -np.pi:
            yr_forward = pivot[1] - radius * np.sin(np.radians(angles))
        # if alpha == -np.pi/2 or alpha == 0.0:
        #     yr_forward = pivot[1] + radius * np.sin(np.radians(angles))
        else:
            yr_forward = pivot[1] + radius * np.sin(np.radians(angles))
        xr_goal, yr_goal = generate_lines(xr_forward[-1], yr_forward[-1], goal[0], goal[1], 5)

        wp[-1-i, :, 0] = np.concatenate([xr_forward, xr_goal[1:]])
        wp[-1-i, :, 1] = np.concatenate([yr_forward, yr_goal[1:]])
            
        # Compute left points
        xl_forward = pivot[0] - radius - radius * np.cos(np.radians(angles))
        if alpha == 0.0 or alpha == np.pi or alpha == -np.pi:
            yl_forward = pivot[1] - radius * np.sin(np.radians(angles))
        # if alpha == -np.pi/2 or alpha == 0.0:
        #     yl_forward = pivot[1] + radius * np.sin(np.radians(angles))
        else:
            yl_forward = pivot[1] + radius * np.sin(np.radians(angles))
        xl_goal, yl_goal = generate_lines(xl_forward[0], yl_forward[0], goal[0], goal[1], 5)

        wp[i, :, 0] = np.concatenate([xl_forward[::-1], xl_goal[1:]])
        wp[i, :, 1] = np.concatenate([yl_forward[::-1], yl_goal[1:]])

        # if alpha != np.pi/2 and alpha != -np.pi/2 and alpha != 0.0 and alpha != np.pi:
        wp[-1-i, :, :] = (inv_rot_mat @ wp[-1-i, :, :].T).T  # Rotate back to original frame
        wp[i, :, :] = (inv_rot_mat @ wp[i, :, :].T).T   # Rotate back to original frame

        # Plot
        plt.plot(wp[i, :, 0], wp[i, :, 1], marker='o', linestyle='-', color=colors[i], label=f"Radius {radius}")
        plt.plot(wp[-1-i, :, 0], wp[-1-i, :, 1], marker='o', linestyle='-', color=colors[i], label=f"Radius {radius}")
    
    # Formatting plot
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Sampled Points with Different Radii")
    plt.legend()
    plt.grid()
    plt.show()

    return wp


def main():
    # Run the function with default parameters
    generate_semicircles()

if __name__ == "__main__":
    main()