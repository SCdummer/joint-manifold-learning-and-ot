import numpy as np
import torchdiffeq
import torch
import os
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from PIL import Image


def create_gaussian(center_x, center_y, side_length, theta, resolution=(28, 28), use_smooth_function=True):
    r"""Creates an image of a gaussian on a 2D plane.

    Args:
        center_x (torch tensor of size (N,)): the x-coordinate of the center of the gaussian.

        center_y (torch tensor of size (N,)): the y-coordinate of the center of the gaussian.

        side_length (torch tensor of size (N,)): the spread of the gaussian.

        theta (torch tensor of size (N,)): the rotation angle for the gaussian.

        resolution (tuple of ints): the resolution of the resulting image. The values of center_x, center_y, and
                                    side_length should all be within [0, resolution[0]], [0, resolution[1]], and
                                    [0, min(resolution[0], resolution[1])] respectively.

        use_smooth_function (bool): whether to create a smooth Gaussian with values slowly decaying towards zero at the
                                    boundary or whether to have an indicator function representing the Gaussian.

    Returns:
        a torch tensor of size (N, resolution[0], resolution[1]) representing images of gaussians.
    """

    # Get a meshgrid
    _x, _y = torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]))

    # Apply the inverse rotation matrix to the points
    xy_centered = torch.stack([_x[None, ...] - center_x[:, None, None], _y[None, ...] - center_y[:, None, None]], dim=1)

    # Apply the inverse rotation matrix to the points
    rotMatrix = torch.zeros(theta.shape[0], 2, 2)
    rotMatrix[:, 0, 0] = torch.cos(theta)
    rotMatrix[:, 0, 1] = -torch.sin(theta)
    rotMatrix[:, 1, 0] = torch.sin(theta)
    rotMatrix[:, 1, 1] = torch.cos(theta)
    xy_rotated = torch.einsum('bij,bjgf->bigf', rotMatrix, xy_centered)

    # Get the gaussian
    side_length = torch.where(side_length / 2 < 1, 1, side_length / 2)
    gaussian_arr = torch.exp(-((xy_rotated[:, 0, ...] / side_length[:, None, None]) ** 2 + (
                xy_rotated[:, 1, ...] / side_length[:, None, None]) ** 2))

    # truncate the gaussian to be zero outside the side_length in a circle
    mask = (xy_rotated[:, 0, ...] ** 2 + xy_rotated[:, 1, ...] ** 2) <= side_length[:, None, None] ** 2
    gaussian_arr = gaussian_arr * mask
    min_inside_gaussian = gaussian_arr[mask].min()
    max_inside_gaussian = gaussian_arr[mask].max()
    gaussian_arr = (gaussian_arr - min_inside_gaussian) / (max_inside_gaussian - min_inside_gaussian)
    gaussian_arr[gaussian_arr < 0] = 0

    # Return the gaussian
    return gaussian_arr


class GaussianTimeSeries:
    r"""A class that generates time series of Gaussians.

    Args:
        use_forcing (bool): a boolean value that indicates whether we should apply some forcing that forces the Gaussians
        to move away from the image boundary.

        shift (int or float): how far away from the actual boundary we should apply the forcing already to make sure
        the Gaussians do not move outside the boundary.

        force_field_magnitude (int or float): the strength of the force term.

        resolution (tuple of ints): resolution of the images.

        use_smooth_function (bool): whether to create a smooth Gaussian with values slowly decaying towards zero at the
                            boundary or whether to have an indicator function representing the Gaussian.

    """

    def __init__(self, use_forcing=True, shift=2, force_field_magnitude=2.0, resolution=(28, 28),
                 use_smooth_function=True):
        self.use_forcing = use_forcing
        self.shift = shift
        if use_forcing:
            self.force_field_magnitude = force_field_magnitude
        else:
            self.force_field_magnitude = 0.0
        self.resolution = resolution
        self.resolution_x, self.resolution_y = resolution
        self.use_smooth_function = use_smooth_function

    def calculate_largest_point(self, gaussian):
        r"""Given a Gaussian image, it calculates the maximum x value, maximum y value, minimum x value, and minimum y
            value of points that lie inside the Gaussian. In other words, this function obtains the bounds of the Gaussian.

        Args:
            gaussian (torch tensor of size (N, self.resolution_x, self.resolution_y)): the Gaussian images.

        Returns:
            min_x (torch tensor of size (N,)): yields the smallest x-value for all coordinates lying inside the Gaussian.

            max_x (torch tensor of size (N,)): yields the largest x-value for all coordinates lying inside the Gaussian.

            min_y (torch tensor of size (N,)): yields the smallest y-value for all coordinates lying inside the Gaussian.

            max_y (torch tensor of size (N,)): yields the largest y-value for all coordinates lying inside the Gaussian.
        """

        # Get the resolutions in each axis
        resolution_x, resolution_y = self.resolution

        # Get the mask
        mask = (gaussian > 0.0)

        # Get the x-index matrix and the y-index matrix
        x_index = torch.arange(0, resolution_x)[None, :].repeat((resolution_y, 1))
        y_index = torch.arange(0, resolution_y)[:, None].repeat((1, resolution_x))

        # Get the maximum and minimum values
        min_x = (mask * x_index[None, ...] + (1.0 - mask.int()) * 10000).min(dim=1).values.min(dim=1).values
        max_x = (mask * x_index[None, ...]).max(dim=1).values.max(dim=1).values
        min_y = (mask * y_index[None, ...] + (1.0 - mask.int()) * 10000).min(dim=1).values.min(dim=1).values
        max_y = (mask * y_index[None, ...]).max(dim=1).values.max(dim=1).values

        # Return these values
        return min_x, max_x, min_y, max_y

    def calc_dzdt_in_safe_region(self, center_x, center_y, side_length, theta):
        r"""Given z = (center_x, center_y, side_length, theta), this function calculates the vector field dzdt
        BEFORE, possibly, applying a force function pushing the Gaussian away from the boundary of the image.

        Args:

            center_x (torch tensor of size (N,)): the x-coordinate of the center of the Gaussian.

            center_y (torch tensor of size (N,)): the y-coordinate of the center of the Gaussian.

            side_length (torch tensor of size (N,)): the side length of the Gaussian.

            theta (torch tensor of size (N,)): the rotation angle for the Gaussian.

        Returns:
            dzdt (torch tensor of size (N, 4)): the vector field dzdt BEFORE, possibly, applying a force.
        """
        raise NotImplementedError

    def vec_field(self, t, z):
        r"""Given z = (center_x, center_y, side_length, theta), this function calculates the vector field dzdt
        ALSO applying a force function pushing the Gaussian away from the boundary of the image.

        Args:

            t (torch tensor): the time at which we observe the vector field.

            z (torch tensor of size (N,4)): z = (center_x, center_y, side_length, theta)


        Returns:
            dzdt (torch tensor of size (N, 4)): the vector field dzdt ALSO applying a force.
        """

        # Get the meaning from z
        center_x, center_y, side_length, theta = z[:, 0], z[:, 1], z[:, 2], z[:, 3]

        # Get the current Gaussian
        gaussian = create_gaussian(center_x, center_y, side_length, theta, self.resolution, self.use_smooth_function)

        # Get the largest points
        min_x, max_x, min_y, max_y = self.calculate_largest_point(gaussian)

        # Get the derivative with respect to the center
        weighting_x = torch.maximum(torch.sigmoid(-(self.resolution_x - max_x - self.shift)) ** 2,
                                    torch.sigmoid(-(min_x - self.shift)) ** 2)
        weighting_y = torch.maximum(torch.sigmoid(-(self.resolution_y - max_y - self.shift)) ** 2,
                                    torch.sigmoid(-(min_y - self.shift)) ** 2)
        weighting = torch.maximum(weighting_x, weighting_y)

        # Get the full derivative
        dzdt_safe_region = self.calc_dzdt_in_safe_region(center_x, center_y, side_length, theta)
        weighting_tensor = torch.zeros_like(dzdt_safe_region)
        weighting_tensor[0, :] = weighting_x
        weighting_tensor[1, :] = weighting_y
        weighting_tensor[2, :] = weighting
        dzdt = dzdt_safe_region * (1.0 - weighting_tensor) + weighting_tensor * self.force_field_magnitude

        # Return the derivative
        return dzdt.permute((1, 0))

    def calculate_image_path(self, center0_x, center0_y, side_length0, theta0, num_time_points, nabla_t):
        r"""Given an initial Gaussian, calculate a time series path using the defined ODE.

        Args:

            center0_x (torch tensor of size (N,)): the x-coordinate of the center of the initial Gaussian.

            center0_y (torch tensor of size (N,)): the y-coordinate of the center of the initial Gaussian.

            side_length0 (torch tensor of size (N,)): the side length of the initial Gaussian.

            theta0 (torch tensor of size (N,)): the rotation angle for the initial Gaussian.

            num_time_points (int): the number of time points in the time series

            nabla_t (float): the time between two timepoints.

        Returns:
            z_t (torch tensor of size (num_time_points, N, 4)): the latent vectors (center_x, center_y, side_length, theta).
                                                                of the Gaussians over time.

            gaussians (torch tensor of size (num_time_points, N, resolution_x, resolution_y): batch of Gaussian image time series.
        """

        # Get the time
        t = np.linspace(0.0, nabla_t * num_time_points, num_time_points + 1)

        # Get the things over time
        y0 = torch.from_numpy(np.stack([center0_x, center0_y, side_length0, theta0], axis=-1)).float()
        t = torch.from_numpy(t).float()
        z_t = torchdiffeq.odeint(self.vec_field, y0=y0, t=t, method='dopri5', atol=5e-4, rtol=1e-4)

        # Get the Gaussians belonging to each of the options
        centert_x, centert_y, side_lengtht, thetat = z_t[..., 0], z_t[..., 1], z_t[..., 2], z_t[..., 3]
        centert_x = centert_x.reshape(-1)
        centert_y = centert_y.reshape(-1)
        side_lengtht = side_lengtht.reshape(-1)
        thetat = thetat.reshape(-1)
        gaussians = create_gaussian(centert_x, centert_y, side_lengtht, thetat, self.resolution,
                                  self.use_smooth_function).reshape(z_t.size(0), z_t.size(1), self.resolution_x,
                                                                    self.resolution_y)

        # Return the Gaussians and the latent vectors generating the Gaussians
        return z_t, gaussians


class GaussianTimeSeriesRandom(GaussianTimeSeries):
    r"""A subclass of the GaussianTimeSeries class that generates time series of Gaussians. These time series are forced
    away from the image boundary via a forcing function inside the ordinary differential equations. The dynamics
    without the forcing applied corresponds to a random linear system of the form:
    :math:`\frac{dz}{dt}=Az`
    with A a random 4x4 matrix and z = (center_x, center_y, side_length, theta) (where [center_x, center_y] is the
    center of the Gaussian, side_length is the side_length of the Gaussian, and theta is the rotation angle).

    Args:
        shift (int or float): how far away from the actual boundary we should apply the forcing already to make sure
        the Gaussians do not move outside the boundary.

        force_field_magnitude (int or float): the strength of the force term.

        resolution (tuple of ints): resolution of the images.

        use_smooth_function (bool): whether to create a smooth Gaussian with values slowly decaying towards zero at the
                            boundary or whether to have an indicator function representing the Gaussian.
    """

    def __init__(self, shift=2, force_field_magnitude=2.0, resolution=(28, 28), use_smooth_function=True):
        super().__init__(use_forcing=True, shift=shift, force_field_magnitude=force_field_magnitude,
                         resolution=resolution, use_smooth_function=use_smooth_function)
        self.dyn_mat = torch.randn(4, 4)

    def calc_dzdt_in_safe_region(self, center_x, center_y, side_length, theta):
        dzdt_safe_region = self.dyn_mat @ torch.stack([center_x, center_y, side_length, theta], dim=0)
        return dzdt_safe_region


class GaussianTimeSeriesDesiredDestination(GaussianTimeSeries):
    r"""A subclass of the GaussianTimeSeries class that generates time series of Gaussians. These time series are
    NOT NECESSARILY forced away from the image boundary via a forcing function inside the ordinary differential
    equations. To see how the dynamics is defined, we suggest looking at the code below. The basis idea is that we want
    to move an initial Gaussian to a predefined final Gaussian. This final Gaussian is the same for each initial Gaussian.
    Note that in this code, the Gaussians all have theta=0.

    Args:
        use_forcing (bool): a boolean value that indicates whether we should apply some forcing that forces the Gaussians
        to move away from the image boundary.

        shift (int or float): how far away from the actual boundary we should apply the forcing already to make sure
        the Gaussians do not move outside the boundary.

        force_field_magnitude (int or float): the strength of the force term.

        resolution (tuple of ints): resolution of the images.

        desired_center_x (torch tensor of size (N,)): the x-coordinate of the Gaussian we want the initial Gaussian to move to.

        desired_center_y (torch tensor of size (N,)): the y-coordinate of the Gaussian we want the initial Gaussian to move to.

        desired_side_length (torch tensor of size (N,)): the side length of the Gaussian we want the initial Gaussian to move to.

        speed_constant_x (float): how fast center_x moves towards its desired value.

        speed_constant_y (float): how fast center_y moves towards its desired value.

        speed_constant_side_length (float): how fast the Gaussians side_length moves towards its desired value.

        approx_const_grad_vel (bool): this determines the form of the ordinary differential equations for. If set to true,
        the ODE is of the form dz_i/dt = -(z_i - desired_value_i)/(abs(z_i - desired_value_i) + epsilon) (WITHOUT
        THE FORCING TERM ADDED). Else, the ODE is of the form dz_i/dt = -(z_i - desired_value_i) (WITHOUT THE FORCING
        TERM ADDED).

        sigma_x (float): there is a gaussian that calculates:
         :math: `torch.exp(-((center_x - self.resolution_x / 2) / sigma_x) ** 2 - ((center_y - self.resolution_y / 2)/(sigma_y)) ** 2)`
         In regions where this value is close to 1, we amplify the speed constants mentioned earlier by a certain
         amplifier constant called 'amplifier'. Here sigma_x determines the standard deviation in the x-direction.

        sigma_y (float): the same but then for the y-coordinate.

        amplifier (float): how much the speed constants are multiplied by in the region of the aforementioned gaussian.

        use_smooth_function (bool): whether to create a smooth Gaussian with values slowly decaying towards zero at the
                                    boundary or whether to have an indicator function representing the Gaussian.
    """

    def __init__(self, use_forcing, shift=2, force_field_magnitude=2.0, resolution=(28, 28),
                 desired_center_x=None, desired_center_y=None, desired_side_length=None,
                 speed_constant_x=0.02, speed_constant_y=0.02, speed_constant_side_length=0.02,
                 approx_const_grad_vel=True, sigma_x=None, sigma_y=None, amplifier=5.0, use_smooth_function=True):
        super().__init__(use_forcing=use_forcing, shift=shift, force_field_magnitude=force_field_magnitude,
                         resolution=resolution, use_smooth_function=use_smooth_function)
        self.desired_center_x = desired_center_x
        self.desired_center_y = desired_center_y
        self.desired_side_length = desired_side_length
        self.speed_constant_x = speed_constant_x
        self.speed_constant_y = speed_constant_y
        self.speed_constant_side_length = speed_constant_side_length
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.amplifier = amplifier
        self.epsilon = min(self.resolution_x, self.resolution_y) / 256
        self.approx_const_grad_vel = approx_const_grad_vel

    def grad_potential(self, x):
        if self.approx_const_grad_vel:
            return (x / (torch.abs(x) + self.epsilon))
        else:
            return x

    def calc_dzdt_in_safe_region(self, center_x, center_y, side_length, theta):

        # Get the full derivative
        weighting_curr_center = (1.0 + self.amplifier * torch.exp(
            -((center_x - self.resolution_x / 2) ** 2 / (self.sigma_x ** 2) + (
                        center_y - self.resolution_y / 2) ** 2 / (self.sigma_y ** 2))))

        dcenter_xdt = -self.grad_potential(
            center_x - self.desired_center_x) * self.speed_constant_x * weighting_curr_center
        dcenter_ydt = -self.grad_potential(
            center_y - self.desired_center_y) * self.speed_constant_y * weighting_curr_center
        dside_lengthdt = -self.grad_potential(
            side_length - self.desired_side_length) * self.speed_constant_side_length * weighting_curr_center
        dthetadt = torch.zeros_like(dcenter_xdt)
        dzdt_safe_region = torch.stack([dcenter_xdt, dcenter_ydt, dside_lengthdt, dthetadt], dim=0)

        return dzdt_safe_region


def create_time_series_gif(image_array, save_dir, time_series_id):
    r"""Saving a gif of the image time series.

    Args:
        image_array (numpy array of size (N, resolution_x, resolution_y): numpy array containing ima single image time series.

        save_dir (Path): path to the directory where we save the gif.

        time_series_id (int): an integer indicating the id of the track / time series
    """

    # Create a figure
    fig, ax = plt.subplots()

    ### trajectory animation (gif)
    def animate(i):
        ax.clear()
        p = ax.imshow(image_array[i].squeeze(), vmin=0, vmax=1)
        ax.set_title("Ground Truth at time point {}".format(i))
        return p,

    gif = FuncAnimation(fig, animate, blit=True, repeat=True, frames=image_array.shape[0], interval=1)
    gif.save(os.path.join(save_dir, "TimeseriesGif_{}.gif".format(time_series_id)), dpi=150,
             writer=PillowWriter(fps=5))
    ax.clear()

    # Close the figures
    plt.close('all')


def save_gaussians_latents_and_gifs(name_gaussians_dataset: str, gaussians: np.ndarray, z_t: np.ndarray):
    r"""Saving a Gaussians dataset.

    Args:
        name_gaussians_dataset (str): a name for the time series

        gaussians (numpy array of size (num_time_points, N, resolution_x, resolution_y)): the Gaussian image time series.

        z_t (numpy array of size (num_time_points, N, 4)): time series of the latent variables z = [center_x, center_y, side_length, theta].
    """

    # Get the directory where we save the Gaussian images as .tif file
    save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", name_gaussians_dataset)

    # Create some directories if they do not exist
    if not os.path.isdir(os.path.join(save_dir, "latents")):
        os.makedirs(os.path.join(save_dir, "latents"))
    if not os.path.isdir(os.path.join(save_dir, "gifs")):
        os.makedirs(os.path.join(save_dir, "gifs"))

    # For every time series
    for i in range(gaussians.shape[1]):

        # We save each time point separately as a .tif file
        for j in range(gaussians.shape[0]):

            # Save the image as a .tif file
            img_obj = Image.fromarray(gaussians[j, i, ...])
            time_label = "00" + str(j) if j < 10 else "0" + str(j)
            save_filename = "track{}_t{}.tif".format(i, time_label)
            dirname = os.path.join(save_dir, "images", "Track{}".format(i))
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            img_obj.save(os.path.join(dirname, save_filename))

        # Save the latent vectors defining the images
        np.save(os.path.join(save_dir, "latents", "Track{}_latent.npy".format(i)), z_t)

        # Finally, save some gifs of the time series
        create_time_series_gif(gaussians[:, i, ...], os.path.join(save_dir, "gifs"), i)


if __name__ == '__main__':
    ###########################################################################
    ### Now defining a time series where Gaussians move from left to right ###
    ###########################################################################

    # Define the number of time series and the number of time points
    num_time_series = 50
    num_time_points = 50

    # Define the resolution in each direction
    resolution_x = 64
    resolution_y = 64

    # Define the time between two time steps
    nabla_t = 0.1

    # Put in the values necessary for generating the dataset
    desired_center_x = resolution_x / 2
    desired_center_y = resolution_y / 8 * 6
    desired_side_length = min(resolution_x, resolution_y) * 5 / 10
    force_field_magnitude = 20.0 * min(resolution_x, resolution_y) / 128
    speed_constant_x = 0.0 * resolution_x / 128
    speed_constant_y = 7.5 * resolution_y / 128
    speed_constant_side_length = 1.5 * min(resolution_x, resolution_y) / 128
    approx_const_grad_vel = True
    sigma_x = max(resolution_x, resolution_y) * 1000
    sigma_y = min(resolution_y, resolution_x) / 8
    amplifier = 20.0 * min(resolution_x, resolution_y) / 128
    use_smooth_function = False
    name_gaussians_dataset = "Gaussian"

    # Define the resolution
    resolution = (resolution_x, resolution_y)

    # Get the object that can be used to create time series
    time_series_creator_destination = GaussianTimeSeriesDesiredDestination(use_forcing=False,
                                                                         shift=int(min(resolution_x, resolution_y) / 8),
                                                                         force_field_magnitude=force_field_magnitude,
                                                                         resolution=resolution,
                                                                         desired_center_x=desired_center_x,
                                                                         desired_center_y=desired_center_y,
                                                                         desired_side_length=desired_side_length,
                                                                         speed_constant_x=speed_constant_x,
                                                                         speed_constant_y=speed_constant_y,
                                                                         speed_constant_side_length=speed_constant_side_length,
                                                                         approx_const_grad_vel=approx_const_grad_vel,
                                                                         sigma_x=sigma_x, sigma_y=sigma_y,
                                                                         amplifier=amplifier,
                                                                         use_smooth_function=use_smooth_function)

    # Get some random initial conditions
    center0_x = resolution_x / 2 + resolution_x / 4 * (2.0 * np.random.rand(num_time_series) - 1)
    center0_y = resolution_y / 6 + 0.0 * center0_x
    side_length0 = min(resolution_x, resolution_y) / 9 + min(resolution_x, resolution_y) / 9 * np.random.rand(
        num_time_series)
    theta0 = 0.0 * (2 * np.random.rand(num_time_series) - 1)

    # Generate random Gaussian time series
    z_t, gaussians = time_series_creator_destination.calculate_image_path(center0_x, center0_y, side_length0, theta0,
                                                                        num_time_points, nabla_t)

    # Save the Gaussians and gifs
    save_gaussians_latents_and_gifs(name_gaussians_dataset, gaussians.numpy(), z_t.numpy())
