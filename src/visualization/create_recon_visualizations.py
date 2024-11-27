# Import plotting libraries
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Import libraries for defining paths
import os

def create_time_series_gif(time_series_gt, time_series_recon, save_dir, recon_idx, base_plot_name: str):

    # Create a figure
    fig, ax = plt.subplots(1, 2)

    # Calculate the maximum value in the ground truth time series
    max_val = time_series_gt.max()

    ### trajectory animation (gif)
    def animate(i, time_series_gt, time_series_recon):
        ax[0].clear()
        ax[1].clear()
        p1 = ax[0].imshow(time_series_gt[i].squeeze(), vmin=0, vmax=max_val)
        p2 = ax[1].imshow(time_series_recon[i].squeeze(), vmin=0, vmax=max_val)
        ax[0].set_title("Ground truth at time point {}".format(i))
        ax[1].set_title("Reconstruction at time point {}".format(i))
        return p1, p2

    gif = FuncAnimation(fig, animate, fargs=(time_series_gt, time_series_recon),
                        blit=True, repeat=True, frames=time_series_gt.shape[0], interval=1)
    gif.save(os.path.join(save_dir, base_plot_name + "_{}.gif".format(recon_idx)), dpi=150, writer=PillowWriter(fps=5))
    ax[0].clear()
    ax[1].clear()

    # Close the figures
    plt.close('all')