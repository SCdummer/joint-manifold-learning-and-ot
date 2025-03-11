import numpy as np
import torch
import matplotlib.pyplot as plt

import os
import sys
import re

main_dir = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(os.path.join(main_dir, "src", "data"))

from create_synthetic_data import LpTimeSeriesLimitCyclesAndEquilibria, create_p_norm_level_set

def plot_latent_vectors(file_path, resolution, big_marker_delta_t, small_marker_delta_t, marker_color='grey'):
    # Load latent vectors from .npy file
    latent_vectors = np.load(file_path)
    
    # If we accidentally saved all latent vectors for all different tracks into one .npy file, we need to extract the latent vectors for the track we want to plot
    if len(latent_vectors.shape) == 3:
        # Get the index of the track
        latent_vectors = latent_vectors[:, int(re.search(r'\d+', os.path.basename(file_path)).group()), ...]
        
    # Transform the first latent vector to a torch tensor
    initial_latent_code = torch.tensor(latent_vectors[0], dtype=torch.float32)
    
    # Create an instance of the LpTimeSeriesLimitCyclesAndEquilibria class
    lp_time_series = LpTimeSeriesLimitCyclesAndEquilibria(shift=resolution/16,
                                                          force_field_magnitude=2.0,
                                                          resolution=(resolution, resolution),
                                                          smooth_function="molifier")
    
    # Save each component of the initial condition in its own variable
    center0_x, center0_y, side_length0, theta0, m0, p0 = torch.unbind(initial_latent_code)
    center0_x = center0_x.unsqueeze(0)
    center0_y = center0_y.unsqueeze(0)
    side_length0 = side_length0.unsqueeze(0)
    theta0 = theta0.unsqueeze(0)
    m0 = m0.unsqueeze(0)
    p0 = p0.unsqueeze(0)
    
    # Get the latent trajectory
    final_time = 5
    delta_t = 0.0001
    num_time_points = int(final_time / delta_t)
    trajectory, _, _ = lp_time_series.calculate_image_path(center0_x, center0_y, side_length0, theta0, m0, p0, num_time_points=num_time_points, nabla_t=delta_t, noise_stds=[])
    trajectory = trajectory.squeeze()
    
    # Calculate the rate in which we plot the big markers and the small markers
    big_marker_rate = big_marker_delta_t * num_time_points
    small_marker_rate = small_marker_delta_t * num_time_points
    
    # Convert the trajectory back to numpy for plotting
    trajectory = trajectory.detach().numpy()
    
    # Only grab the relevant parameters and normalize the side length between 0 and 4
    l, p = trajectory[:, 2], trajectory[:, 5]
    l = l / resolution * 4
    
    # Create plot
    fig, ax = plt.subplots()
    
    # Plot lines between points
    ax.plot(p, l, linestyle='--', color=marker_color, alpha=0.5)
    
    # Plot markers with labels
    for i in range(trajectory.shape[0]):
        if i % big_marker_rate == 0:
            ax.plot(p[i], l[i], 'o', markersize=6, color=marker_color)
            #ax.text(p[i] - 0.05, l[i] + 0.05, f't={i*delta_t:.2f}', fontsize=8, ha='right', va='bottom')
        elif i % small_marker_rate == 0:
            ax.plot(p[i], l[i], 'o', markersize=3, color=marker_color)
            #ax.text(p[i] - 0.05, l[i] + 0.05, f't={i*delta_t:.2f}', fontsize=8, ha='right', va='bottom')
    
    # Set x and y limits if provided
    ax.set_xlim([0.0, 4.0])
    ax.set_ylim([0.0, 4.0])
    
    # Add an arrow with circles and y-label
    fig.subplots_adjust(left=0.2, bottom=0.2)
    arrow_ax = fig.add_axes([0.075, 0.1, 0.05, 0.8], frameon=False)
    arrow_ax.set_xlim([0, 1])
    arrow_ax.set_ylim([0, 1])
    
    arrow_ax.annotate("", xytext=(1.0, 0.15), xy=(1.0, 0.95),
                arrowprops=dict(arrowstyle="->"))
    arrow_ax.plot(0.5, 0.175, 'o', markersize=3, color='k')
    arrow_ax.plot(0.5, 0.9, 'o', markersize=15, color='k')
    arrow_ax.text(0.5, 0.55, 'Size', fontsize=12, ha='center', va='center', rotation='vertical')
    arrow_ax.axis('off')
    
    # Add an arrow with shapes and x-label
    arrow_ax_x = fig.add_axes([0.2, 0.1, 0.7, 0.05], frameon=False)
    arrow_ax_x.set_xlim([0, 4])
    arrow_ax_x.set_ylim([0, 1])
    
    arrow_ax_x.annotate("", xytext=(0.05, 0.5), xy=(3.95, 0.5),
                arrowprops=dict(arrowstyle="->"))
    
    # Generate shapes
    resolution = 128
    center_x = torch.tensor([resolution // 2]).float()
    center_y = torch.tensor([resolution // 2]).float()
    side_length = torch.tensor([resolution]).float() * 0.75
    theta = torch.tensor([0.0]).float()
    m = torch.tensor([1.0]).float()
    
    size = 0.065
    ax_shape1 = fig.add_axes([0.2, 0.01, size, size], frameon=False)
    ax_shape2 = fig.add_axes([0.55 - 0.5*size, 0.01, size, size], frameon=False)
    ax_shape3 = fig.add_axes([0.9 - size, 0.01, size, size], frameon=False)
    # ax_shape1.set_xlim([0, 1])
    # ax_shape1.set_ylim([0, 1])
    # ax_shape2.set_xlim([0, 1])
    # ax_shape2.set_ylim([0, 1])
    # ax_shape3.set_xlim([0, 1])
    # ax_shape3.set_ylim([0, 1])
    
    shape1 = create_p_norm_level_set(center_x, center_y, side_length, theta, m, torch.tensor([0.5]), resolution=(resolution, resolution))
    shape2 = create_p_norm_level_set(center_x, center_y, side_length, theta, m, torch.tensor([2.0]), resolution=(resolution, resolution))
    shape3 = create_p_norm_level_set(center_x, center_y, side_length, theta, m, torch.tensor([10.0]), resolution=(resolution, resolution))

    ax_shape1.imshow(~shape1[0], cmap='gray')
    ax_shape2.imshow(~shape2[0], cmap='gray')
    ax_shape3.imshow(~shape3[0], cmap='gray')
    
    ax_shape1.axis('off')
    ax_shape2.axis('off')
    ax_shape3.axis('off')
    
    arrow_ax_x.text(2, 0.0, 'Shape', fontsize=12, ha='center', va='center')
    arrow_ax_x.axis('off')
    
    # Show plot
    ax.set_title('Ground truth image parameters over time')
    plt.xlabel('Shape')
    plt.ylabel('Size')
    plt.show()

# Example usage
file_path = os.path.join(main_dir, "data", "LimitCyclesAndEquilibria", "latents", "Track0_latent.npy")
big_marker_rate = 0.1
small_marker_rate = 0.05
plot_latent_vectors(file_path, 64, big_marker_rate, small_marker_rate, marker_color='grey')
