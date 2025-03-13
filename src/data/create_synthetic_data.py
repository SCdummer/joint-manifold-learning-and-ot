'''
Currently, this .py file is not used for producing the results in the paper.
It is a generalized version of the create_gaussian_data.py code as it also allows to create images of other p-level sets (instead of the Gaussian 2-level set).
Moreover, it contains one additional time series that has some limit cycle behavior. 
'''

import numpy as np
import torchdiffeq
import torch
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from PIL import Image



######################################################################################################################
### Function for generating images of squares, gaussians/circles, and general interiors of ||.||_p norm level sets ###
######################################################################################################################



def create_p_norm_level_set(center_x, center_y, side_length, theta, m, p, resolution=(28, 28), smooth_function=None):
    r"""Creates an image of the set ||R_theta @ ((x,y) - (center_x, center_y))||_p <= side_length/2 with R_theta the rotation matrix with angle theta.
    Inside the level set a Gaussian/mollifier-like function is plotted.

    Args:
        center_x (torch tensor of size (N,)): the x-coordinate of the center.

        center_y (torch tensor of size (N,)): the y-coordinate of the center.

        side_length (torch tensor of size (N,)): the 'width' of the set ||R_theta @ ((x,y) - (center_x, center_y))||_p <= side_length/2.

        theta (torch tensor of size (N,)): the rotation angle for the set.
        
        m (torch tensor of size (N,)): the mass of the set. In other words, the value with which the Gaussian/mollifier-like function is multiplied.
        
        p (torch tensor of size (N,)): the p-value for the lp norm.

        resolution (tuple of ints): the resolution of the resulting image. The values of center_x, center_y, and
                                    side_length should all be within [0, resolution[0]], [0, resolution[1]], and
                                    [0, min(resolution[0], resolution[1])] respectively.

        smooth_function (str | None): whether to use a Gaussian or mollifier-like function in the interior of the set. 
                                      If smooth_function=None, we just use a segmentation mask. 

    Returns:
        a torch tensor of size (N, resolution[0], resolution[1]) representing the images.
    """

    # Get a meshgrid
    _x, _y = torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]))

    # Center the points
    xy_centered = torch.stack([_x[None, ...] - center_x[:, None, None], _y[None, ...] - center_y[:, None, None]], dim=1)

    # Apply the inverse rotation matrix to the points
    rotMatrix = torch.zeros(theta.shape[0], 2, 2)
    rotMatrix[:, 0, 0] = torch.cos(theta)
    rotMatrix[:, 0, 1] = -torch.sin(theta)
    rotMatrix[:, 1, 0] = torch.sin(theta)
    rotMatrix[:, 1, 1] = torch.cos(theta)
    xy_rotated = torch.einsum('bij,bjgf->bigf', rotMatrix, xy_centered)

    # Define the resolutions
    resolution_x, resolution_y = resolution

    # Create a nice profile for the image using a molifier function or a gaussian
    if smooth_function == "molifier":
        
        # Get the radius
        radius = torch.where(side_length / 2 < 1, 1, side_length / 2) / (max(resolution_x, resolution_y) / 2)
        
        # Coefficient function
        alpha = 0.5
        beta = 10
        sigma = 1 - torch.sigmoid(alpha * (p[:, None, None] - beta))
            
        # Getting the interior function values
        shifted_x = xy_rotated[:, 0, ...] / (resolution_x / 2)
        shifted_y = xy_rotated[:, 1, ...] / (resolution_y / 2)
        xy_squared_lp_norm = sigma * (torch.abs(shifted_x) ** p[:, None, None] + torch.abs(shifted_y) ** p[:, None, None]) ** (1 /  p[:, None, None])
        xy_squared_lp_norm += (1 - sigma) * torch.maximum(torch.abs(shifted_x), torch.abs(shifted_y))
        scaling = radius[:, None, None]
        molifier_profile = torch.where(xy_squared_lp_norm  - radius[:, None, None] < 0,
                                       torch.exp(-1.0 / (1 - xy_squared_lp_norm / (scaling)) + 1),
                                       0.0 * torch.ones_like(xy_squared_lp_norm))
        molifier_profile = m[:, None, None] * molifier_profile
    elif smooth_function == 'gaussian':
        
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
        
        # Set the molifier_profile variable
        molifier_profile = gaussian_arr
        
    else:
        
        # Get the radius
        radius = torch.where(side_length / 2 < 1, 1, side_length / 2) / (max(resolution_x, resolution_y) / 2)
        
        # Coefficient function
        alpha = 0.5
        beta = 10
        sigma = 1 - torch.sigmoid(alpha * (p[:, None, None] - beta))
            
        # Getting the interior function values
        shifted_x = xy_rotated[:, 0, ...] / (resolution_x / 2)
        shifted_y = xy_rotated[:, 1, ...] / (resolution_y / 2)
        xy_squared_lp_norm = sigma * (torch.abs(shifted_x) ** p[:, None, None] + torch.abs(shifted_y) ** p[:, None, None]) ** (1 /  p[:, None, None])
        xy_squared_lp_norm += (1 - sigma) * torch.maximum(torch.abs(shifted_x), torch.abs(shifted_y))
        scaling = radius[:, None, None]
        mask = ((xy_squared_lp_norm  - radius[:, None, None]) <= 0)
        
        # Just equate molifier profile to mask as then we just output the mask
        molifier_profile = mask

    # Return the molifier profile
    return molifier_profile



##################################
### Defining image time series ###
##################################

def cubic_spline_interpolation(x, y, x_new):
    """
    Perform cubic spline interpolation between given points.

    Parameters:
    x (torch.Tensor): Known x-coordinates.
    y (torch.Tensor): Known y-coordinates.
    x_new (torch.Tensor): New x-coordinates to interpolate.

    Returns:
    torch.Tensor: Interpolated y-coordinates at x_new.
    """
    # Ensure x and y are 1D tensors
    x = x.flatten()
    y = y.flatten()

    # Sort x and y based on x
    sorted_indices = torch.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    # Calculate the differences
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]

    # Calculate the slopes
    slopes = dy / dx

    # Set up the tridiagonal system to solve for second derivatives
    n = len(x)
    A = torch.zeros((n, n))
    b = torch.zeros(n)

    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, n - 1):
        A[i, i - 1] = dx[i - 1]
        A[i, i] = 2 * (dx[i - 1] + dx[i])
        A[i, i + 1] = dx[i]
        b[i] = 6 * (slopes[i] - slopes[i - 1])

    # Solve for the second derivatives
    d2y = torch.linalg.solve(A, b)

    # Calculate the coefficients for the cubic spline
    a = y[:-1]
    b = slopes - dx * (2 * d2y[:-1] + d2y[1:]) / 6
    c = d2y[:-1] / 2
    d = (d2y[1:] - d2y[:-1]) / (6 * dx)

    # Interpolate the values
    y_new = torch.zeros_like(x_new)
    for i in range(len(x) - 1):
        mask = (x_new >= x[i]) & (x_new <= x[i + 1])
        dx_new = x_new[mask] - x[i]
        y_new[mask] = a[i] + b[i] * dx_new + c[i] * dx_new**2 + d[i] * dx_new**3

    return y_new

class LpTimeSeries:
    
    r"""A class that generates time series of circles, squares, and other interiors of l_p norms.

    Args:
        use_forcing (bool): a boolean value that indicates whether we should apply some forcing that forces the shapes
        to move away from the image boundary.

        shift (int or float): how far away from the actual boundary we should apply the forcing already to make sure
        the shapes do not move outside the boundary.

        force_field_magnitude (int or float): the strength of the force term.

        resolution (tuple of ints): resolution of the images.

        smooth_function (str | None): whether to use a Gaussian or mollifier-like function in the interior of the set. 
                                      If smooth_function=None, we just use a segmentation mask. 

    """

    def __init__(self, use_forcing=True, shift=2, force_field_magnitude=2.0, resolution=(28, 28),
                 smooth_function=None):
        self.use_forcing = use_forcing
        self.shift = shift
        if use_forcing:
            self.force_field_magnitude = force_field_magnitude
        else:
            self.force_field_magnitude = 0.0
        self.resolution = resolution
        self.resolution_x, self.resolution_y = resolution
        self.smooth_function = smooth_function
        
        self.cubic_spline_x = torch.tensor([0.0, 1.0, 3.0, 4.0]) #torch.tensor([0.0, 1.0, 3.0, 4.0])
        self.cubic_spline_y = torch.tensor([0.0, 0.65, 4.0, 10.0]) #torch.tensor([0.0, 1.0, 4.0, 10.0])

    def calculate_largest_point(self, shape_img):
        r"""Given a generated image belonging to some l_p level set, it calculates the maximum x value, maximum y value, minimum x value, and minimum y
            value of points that lie inside the l_p level set. In other words, this function obtains the bounds / bounding box of the level set.

        Args:
            shape_img (torch tensor of size (N, self.resolution_x, self.resolution_y)): the images.

        Returns:
            min_x (torch tensor of size (N,)): yields the smallest x-value for all coordinates lying inside the shape.

            max_x (torch tensor of size (N,)): yields the largest x-value for all coordinates lying inside the shape.

            min_y (torch tensor of size (N,)): yields the smallest y-value for all coordinates lying inside the shape.

            max_y (torch tensor of size (N,)): yields the largest y-value for all coordinates lying inside the shape.
        """

        # Get the resolutions in each axis
        resolution_x, resolution_y = self.resolution

        # Get the mask
        mask = (shape_img > 0.0)

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

    def calc_dzdt_in_safe_region(self, center_x, center_y, side_length, theta, m, p):
        r"""Given z = (center_x, center_y, side_length, theta), this function calculates the vector field dzdt
        BEFORE, possibly, applying a force function pushing the shape away from the boundary of the image.

        Args:

            center_x (torch tensor of size (N,)): the x-coordinate of the center of the shape.

            center_y (torch tensor of size (N,)): the y-coordinate of the center of the shape.

            side_length (torch tensor of size (N,)): the side length of the shape.

            theta (torch tensor of size (N,)): the rotation angle for the shape.
            
            p (torch tensor of size (N,)): the p value in the l_p norm, which determines the shape of the object.
            
            m (torch tensor of size (N,)): the mass / maximum intensity that the gaussian or molifier

        Returns:
            dzdt (torch tensor of size (N, 4)): the vector field dzdt BEFORE, possibly, applying a force.
        """
        raise NotImplementedError

    def vec_field(self, t, z):
        r"""Given z = (center_x, center_y, side_length, theta), this function calculates the vector field dzdt
        ALSO applying a force function pushing the shape away from the boundary of the image.

        Args:

            t (torch tensor): the time at which we observe the vector field.

            z (torch tensor of size (N,6)): z = (center_x, center_y, side_length, theta, m, p)


        Returns:
            dzdt (torch tensor of size (N, 6)): the vector field dzdt ALSO applying a force.
        """

        # Get the meaning from z
        center_x, center_y, side_length, theta, m, p = z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4], z[:, 5]

        # Get the current shape / image
        shape_img = create_p_norm_level_set(center_x, center_y, side_length, theta, m, cubic_spline_interpolation(self.cubic_spline_x, self.cubic_spline_y, p), resolution=self.resolution, smooth_function=self.smooth_function)

        # Get the largest points
        min_x, max_x, min_y, max_y = self.calculate_largest_point(shape_img)

        # Get the derivative with respect to the center
        weighting_x = torch.maximum(torch.sigmoid(-(self.resolution_x - max_x - self.shift)) ** 2,
                                    torch.sigmoid(-(min_x - self.shift)) ** 2)
        weighting_y = torch.maximum(torch.sigmoid(-(self.resolution_y - max_y - self.shift)) ** 2,
                                    torch.sigmoid(-(min_y - self.shift)) ** 2)
        weighting = torch.maximum(weighting_x, weighting_y)

        # Get the full derivative
        dzdt_safe_region = self.calc_dzdt_in_safe_region(center_x, center_y, side_length, theta, m, p)
        weighting_tensor = torch.zeros_like(dzdt_safe_region)
        weighting_tensor[0, :] = weighting_x
        weighting_tensor[1, :] = weighting_y
        weighting_tensor[2, :] = weighting
        weighting_tensor[5, :] = weighting
        dzdt = dzdt_safe_region * (1.0 - weighting_tensor) + weighting_tensor * self.force_field_magnitude

        # Return the derivative
        return dzdt.permute((1, 0))

    def calculate_image_path(self, center0_x, center0_y, side_length0, theta0, m0, p0, num_time_points=50, nabla_t=0.1, noise_stds=[]):
        r"""Given an initial image, calculate a time series path using the defined ODE.

        Args:

            center0_x (torch tensor of size (N,)): the x-coordinate of the center of the initial shape.

            center0_y (torch tensor of size (N,)): the y-coordinate of the center of the initial shape.

            side_length0 (torch tensor of size (N,)): the side length of the initial shape.

            theta0 (torch tensor of size (N,)): the rotation angle for the initial shape.
            
            m0 (torch tensor of size (N,)): the mass / maximum intensity of the initial shape (only needed when self.smooth_function is NOT None)
            
            p0 (torch tensor of size (N,)): the initial p value in the l_p norm determining the initial shape.

            num_time_points (int): the number of time points in the time series

            nabla_t (float): the time between two timepoints.
            
            noise_stds (list of floats): the standard deviation of the noise added to the image time series.

        Returns:
            z_t (torch tensor of size (num_time_points, N, 6)): the latent vectors (center_x, center_y, side_length, theta, m, p)
                                                                of the shapes over time.

            shape_imgs (torch tensor of size (num_time_points, N, resolution_x, resolution_y): batch of image time series.
        """

        # Get the time points
        t = np.linspace(0.0, nabla_t * num_time_points, num_time_points + 1)

        # Get the things over time
        y0 = torch.from_numpy(np.stack([center0_x, center0_y, side_length0, theta0, m0, p0], axis=-1)).float()
        t = torch.from_numpy(t).float()
        z_t = torchdiffeq.odeint(self.vec_field, y0=y0, t=t, method='dopri5', atol=5e-4, rtol=1e-4)

        # Get the images belonging to each of the options
        centert_x, centert_y, side_lengtht, thetat, mt, pt = z_t[..., 0], z_t[..., 1], z_t[..., 2], z_t[..., 3], z_t[..., 4], z_t[..., 5]
        centert_x = centert_x.reshape(-1)
        centert_y = centert_y.reshape(-1)
        side_lengtht = side_lengtht.reshape(-1)
        thetat = thetat.reshape(-1)
        mt = mt.reshape(-1)
        pt = pt.reshape(-1)
        pt = cubic_spline_interpolation(self.cubic_spline_x, self.cubic_spline_y, pt)
        shape_imgs = create_p_norm_level_set(centert_x, centert_y, side_lengtht, thetat, mt, pt, 
                                             self.resolution, self.smooth_function).reshape(z_t.size(0), z_t.size(1), self.resolution_x, self.resolution_y)

        # Also create a noisy dataset
        shape_imgs_noisy = []
        for noise_std in noise_stds:
            shape_imgs_noisy.append(torch.clamp(shape_imgs + noise_std * torch.randn_like(shape_imgs), 0.0, 1.0))

        # Return the shape images and the latent vectors generating these images.
        return z_t, shape_imgs, shape_imgs_noisy


class LpTimeSeriesRandom(LpTimeSeries):
    r"""A subclass of the LpTimeSeries class that generates time series of shapes. These time series are forced
    away from the image boundary via a forcing function inside the ordinary differential equations. The dynamics
    without the forcing applied corresponds to a random linear system of the form:
    :math:`\frac{dz}{dt}=Az`
    with A a random 6x6 matrix and z = (center_x, center_y, side_length, theta, m, p) (where [center_x, center_y] is the
    center of the shape, side_length is the side_length of the shape, theta is the rotation angle, m is the mass of the image
    , and p determines the l_p norm used).

    Args:
        shift (int or float): how far away from the actual boundary we should apply the forcing already to make sure
        the squares do not move outside the boundary.

        force_field_magnitude (int or float): the strength of the force term.

        resolution (tuple of ints): resolution of the images.

        smooth_function (str | None): whether to use a Gaussian or mollifier-like function in the interior of the set. 
                                      If smooth_function=None, we just use a segmentation mask. 
                                      
        variables_to_keep_fixed (list[str]): list of variables to keep fixed. Possible options are in: ["center_x", "center_y", "side_length", "theta", "m", "p"]
    """

    def __init__(self, shift=2, force_field_magnitude=2.0, resolution=(28, 28), smooth_function=None, variables_to_keep_fixed=[]):
        super().__init__(use_forcing=True, shift=shift, force_field_magnitude=force_field_magnitude,
                         resolution=resolution, smooth_function=smooth_function)
        self.dyn_mat = torch.randn(6, 6)
        for i, variable in enumerate(["center_x", "center_y", "side_length", "theta", "m", "p"]):
            if variable in variables_to_keep_fixed:
                self.dyn_mat[:, i] = 0
                self.dyn_mat[i, :] = 0

    def calc_dzdt_in_safe_region(self, center_x, center_y, side_length, theta, m, p):
        dzdt_safe_region = self.dyn_mat @ torch.stack([center_x, center_y, side_length, theta, m, p], dim=0)
        return dzdt_safe_region


class LpTimeSeriesDesiredDestination(LpTimeSeries):
    r"""A subclass of the LpTimeSeries class that generates time series. These time series are
    NOT NECESSARILY forced away from the image boundary via a forcing function inside the ordinary differential
    equations. To see how the dynamics is defined, we suggest looking at the code below. The basis idea is that we want
    to move an initial shape to a predefined final shape. This final shape is the same for each initial shape.
    Note that in this code, the shape all have theta=0.

    Args:
        use_forcing (bool): a boolean value that indicates whether we should apply some forcing that forces the shapes
        to move away from the image boundary.

        shift (int or float): how far away from the actual boundary we should apply the forcing already to make sure
        the shapes do not move outside the boundary.

        force_field_magnitude (int or float): the strength of the force term.

        resolution (tuple of ints): resolution of the images.

        desired_center_x (torch tensor of size (N,)): the x-coordinate of the shape we want the initial shape to move to.

        desired_center_y (torch tensor of size (N,)): the y-coordinate of the shape we want the initial shape to move to.

        desired_side_length (torch tensor of size (N,)): the side length of the shape we want the initial shape to move to.
        
        desired_m (torch tensor of size (N,)): the mass / maximum intensity of the image IN the shape we want the initial shape to move to.
        
        desired_p (torch tensor of size (N,)): the l_p norm of the shape we want the initial shape to move to.

        speed_constant_x (float): how fast center_x moves towards its desired value.

        speed_constant_y (float): how fast center_y moves towards its desired value.

        speed_constant_side_length (float): how fast the squares side_length moves towards its desired value.
        
        speed_constant_m (float): how fast m moves towards its desired value.
        
        speed_constant_p (float): how fast p moves towards its desired value.

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
        
        smooth_function (str | None): whether to use a Gaussian or mollifier-like function in the interior of the set. 
                                      If smooth_function=None, we just use a segmentation mask. 
    """

    def __init__(self, use_forcing, shift=2, force_field_magnitude=2.0, resolution=(28, 28),
                 desired_center_x=None, desired_center_y=None, desired_side_length=None, desired_m=None, desired_p=None, 
                 speed_constant_x=0.02, speed_constant_y=0.02, speed_constant_side_length=0.02, speed_constant_m=None, speed_constant_p=None,
                 approx_const_grad_vel=True, sigma_x=None, sigma_y=None, amplifier=5.0, smooth_function=None):
        super().__init__(use_forcing=use_forcing, shift=shift, force_field_magnitude=force_field_magnitude,
                         resolution=resolution, smooth_function=smooth_function)
        self.desired_center_x = desired_center_x
        self.desired_center_y = desired_center_y
        self.desired_side_length = desired_side_length
        self.desired_m = desired_m
        self.desired_p = desired_p
        self.speed_constant_x = speed_constant_x
        self.speed_constant_y = speed_constant_y
        self.speed_constant_side_length = speed_constant_side_length
        self.speed_constant_m = speed_constant_m
        self.speed_constant_p = speed_constant_p
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

    def calc_dzdt_in_safe_region(self, center_x, center_y, side_length, theta, m, p):

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
        dmdt = -self.grad_potential(
            m - self.desired_m) * self.speed_constant_m * weighting_curr_center
        dpdt = -self.grad_potential(
            p - self.desired_p) * self.speed_constant_p * weighting_curr_center
        dzdt_safe_region = torch.stack([dcenter_xdt, dcenter_ydt, dside_lengthdt, dthetadt, dmdt, dpdt], dim=0)

        return dzdt_safe_region

class LpTimeSeriesRegulatedGrowth(LpTimeSeries):
    r"""A subclass of the LpTimeSeries class that generates time series of shapes. These time series are forced
    away from the image boundary via a forcing function inside the ordinary differential equations. The dynamics
    without the forcing applied corresponds to a system of the form

    .. math::
       \\frac{dz_1}{dt} = 0 

    .. math::
       \\frac{dz_2}{dt} = 0 

    .. math::
       \\frac{dz_3}{dt} = z_3\\left(\\alpha - \\beta z_3 - \\gamma (z_5 + z_6)\\right)
    
    .. math::
       \\frac{dz_4}{dt} = 0
       
     .. math::
       \\frac{dz_5}{dt} = z_5\\left(\\delta z_3 - \\epsilon z_5\\right)

    .. math::
       \\frac{dz_6}{dt} = z_6\\left(\\delta z_3 - \\epsilon z_6\\right)
    
    with z = (center_x, center_y, side_length, theta, m, p) (where [center_x, center_y] is the
    center of the shape, side_length is the side_length of the shape, theta is the rotation angle, m is the mass of the image
    , and p determines the l_p norm used).

    Args:
        shift (int or float): how far away from the actual boundary we should apply the forcing already to make sure
        the squares do not move outside the boundary.

        force_field_magnitude (int or float): the strength of the force term.

        resolution (tuple of ints): resolution of the images.

        smooth_function (str | None): whether to use a Gaussian or mollifier-like function in the interior of the set. 
                                      If smooth_function=None, we just use a segmentation mask. 
                                      
        alpha (float): the constant alpha used in the above system of equations
        
        beta (float): the constant beta used in the above system of equations
        
        gamma (float): the constant gamma used in the above system of equations
        
        delta (float): the constant delta used in the above system of equations
        
        epsilon (float): the constant epsilon used in the above system of equations
    """

    def __init__(self, shift=2, force_field_magnitude=2.0, resolution=(28, 28), smooth_function=None, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, epsilon=1.0):
        super().__init__(use_forcing=False, shift=shift, force_field_magnitude=force_field_magnitude,
                         resolution=resolution, smooth_function=smooth_function)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
    
    def calc_dzdt_in_safe_region(self, center_x, center_y, side_length, theta, m, p): 
        # dmdt = m * (self.alpha - self.beta * m - self.gamma * (p + side_length))
        # dpdt = p * (self.delta * m - self.epsilon * p)
        # dside_lengthdt = side_length * (self.delta * m - self.epsilon * side_length)
        dcenter_xdt = torch.zeros_like(m)
        dcenter_ydt = torch.zeros_like(dcenter_xdt)
        dmdt = torch.zeros_like(dcenter_xdt)
        
        # The Lorenz system with chaotic parameters       
        theta_lbound = -2.0*torch.pi
        theta_ubound = 2.0*torch.pi
        p_lbound = 0.0
        p_ubound = 4.5
        l_lbound = 0.05
        l_ubound = 0.8
        
        l = side_length / (max(self.resolution))
        theta_rescaled = (theta - theta_lbound) / (theta_ubound - theta_lbound) 
        p_rescaled = (p - p_lbound) / (p_ubound - p_lbound)
        l_rescaled = (l - l_lbound) / (l_ubound - l_lbound)
        
        # Calculate the Lorenz system on the scaled state and subsequently scaling back again
        name_val_pair = {"z": ("theta", theta_rescaled), 
                         "y": ("p", p_rescaled), 
                         "x": ("l", l_rescaled)}
        x = name_val_pair["x"][-1] * 60.0 - 30.0
        y = name_val_pair["y"][-1] * 60.0 - 30.0
        z = name_val_pair["z"][-1] * 50.0
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        
        dxdt = 1 / 60.0 * dxdt
        dydt = 1 / 60.0 * dydt
        dzdt = 1 / 50.0 * dzdt
        
        # Assigning the correct values to the correct thing
        for dt, state in zip([dxdt, dydt, dzdt], ["x","y","z"]):
            if name_val_pair[state][0] == "theta":
                dthetadt = dt
            elif name_val_pair[state][0] == "p":
                dpdt = dt
            else:
                dldt = dt
        
        # Do some final rescaling, back to the original values
        dthetadt = (theta_ubound - theta_lbound) * dthetadt
        dpdt = (p_ubound - p_lbound) * dpdt
        dside_lengthdt = (l_ubound - l_lbound) * max(self.resolution) * dldt

        
        # sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        # dmdt = sigma * (p - m)
        # dpdt = m * (rho - side_length) - p
        # dside_lengthdt = m * p - beta * side_length
        
        # dmdt = dmdt * torch.sigmoid(2.0 * (1.0 - m))
        # dpdt = dpdt * (torch.sigmoid(2.0 * (1.0 - p)) + torch.sigmoid(2.0 * (p - 0.05)))
        # dside_lengthdt = dside_lengthdt * (torch.sigmoid(2.0 * (0.5 - side_length)))
        
        return torch.stack([dcenter_xdt, dcenter_ydt, dside_lengthdt, dthetadt, dmdt, dpdt], dim=0)


class LpTimeSeriesLimitCyclesAndEquilibria(LpTimeSeries):
    r"""A subclass of the LpTimeSeries class that generates time series of shapes. These time series are forced
    away from the image boundary via a forcing function inside the ordinary differential equations. The dynamics
    without the forcing applied corresponds to a system of the form

    .. math::
       \\frac{dz_1}{dt} = 0 

    .. math::
       \\frac{dz_2}{dt} = 0 

    .. math::
       \\frac{dz_3}{dt} = z_3\\left(\\alpha - \\beta z_3 - \\gamma (z_5 + z_6)\\right)
    
    .. math::
       \\frac{dz_4}{dt} = 0
       
     .. math::
       \\frac{dz_5}{dt} = z_5\\left(\\delta z_3 - \\epsilon z_5\\right)

    .. math::
       \\frac{dz_6}{dt} = z_6\\left(\\delta z_3 - \\epsilon z_6\\right)
    
    with z = (center_x, center_y, side_length, theta, m, p) (where [center_x, center_y] is the
    center of the shape, side_length is the side_length of the shape, theta is the rotation angle, m is the mass of the image
    , and p determines the l_p norm used).

    Args:
        shift (int or float): how far away from the actual boundary we should apply the forcing already to make sure
        the squares do not move outside the boundary.

        force_field_magnitude (int or float): the strength of the force term.

        resolution (tuple of ints): resolution of the images.

        smooth_function (str | None): whether to use a Gaussian or mollifier-like function in the interior of the set. 
                                      If smooth_function=None, we just use a segmentation mask. 
                                      
        alpha (float): the constant alpha used in the above system of equations
        
        beta (float): the constant beta used in the above system of equations
        
        gamma (float): the constant gamma used in the above system of equations
        
        delta (float): the constant delta used in the above system of equations
        
        epsilon (float): the constant epsilon used in the above system of equations
    """

    def __init__(self, shift=2, force_field_magnitude=2.0, resolution=(28, 28), smooth_function=None):
        super().__init__(use_forcing=False, shift=shift, force_field_magnitude=force_field_magnitude,
                         resolution=resolution, smooth_function=smooth_function)
        
        # Define some parameters for scaling the side length
        self.scaling_factor = 4.0
        
        # Defining the seperatrix starting height
        self.start_height = 0.75
        self.end_height = 1.5
        
        # Some parameters for defining the A1 and A2 matrix and the general shape of the level set
        v_1_11 = 0.0
        v_2_11 = 1.5
        scale_1_1 = .50
        scale_1_2 = 0.05
        scale_2_1 = 0.05 #0.15
        scale_2_2 = 0.5 #1.5
        self.d = 0.005
        self.k = 10.0
        self.a = 30.0
        self.level_set_value = 0.1 / 3.0
        self.only_region_1 = True
        
        # Define centers for the F function
        if self.only_region_1:
            self.c1 = torch.tensor([1.0, 2.5])
            self.c2 = torch.tensor([2.25, 1.5])
        else:
            self.c1 = torch.tensor([1.0, 2.5])
            self.c2 = torch.tensor([2.25, 1.75])
        #self.c2 = torch.tensor([2.0, 1.5])
        
        # Define the matrix A1
        v_1 = torch.tensor([1, v_1_11])
        c = -1.0 / v_1[1] if v_1[1] != 0 else 0.0
        v_2 = torch.tensor([c, 1])
        v_1 = v_1 / torch.norm(v_1)
        v_2 = v_2 / torch.norm(v_2)
        self.A1 = (scale_1_1 * torch.outer(v_1, v_1) +  scale_1_2 * torch.outer(v_2, v_2))

        # Define the matrix A2
        v_1 = torch.tensor([1, v_2_11])
        c = -1.0 / v_1[1] if v_1[1] != 0 else 0.0
        v_2 = torch.tensor([c, 1])
        v_1 = v_1 / torch.norm(v_1)
        v_2 = v_2 / torch.norm(v_2)
        self.A2 = (scale_2_1 * torch.outer(v_1, v_1) +  scale_2_2 * torch.outer(v_2, v_2))
        
        # Get the coordinates of the equilibrium
        self.equilibrium_side_length = 0.2 * max(self.resolution)
        self.equilibrium_p = 1.0
        
    def F(self, X, Y):
        XY = torch.stack((X, Y), axis=-1)
        XY_r = XY.reshape(-1, 2)
        F1 = (torch.sum((XY_r - self.c1) * ((XY_r - self.c1) @ self.A1.T), dim=-1)).reshape(*XY.shape[:-1]) + self.d * torch.sin(self.k * (X - self.c1[0])) * torch.cos(self.k * (Y - self.c1[1]))
        F2 = (torch.sum((XY_r - self.c2) * ((XY_r - self.c2) @ self.A2.T), dim=-1)).reshape(*XY.shape[:-1]) + self.d * torch.sin(self.k * (X - self.c2[0])) * torch.cos(self.k * (Y - self.c2[1]))
        weight_factor_1 = torch.exp(-self.a * torch.sum((XY_r - self.c1) * ((XY_r - self.c1) @ self.A1.T), dim=-1)).reshape(*XY.shape[:-1])
        weight_factor_2 = torch.exp(-self.a * torch.sum((XY_r - self.c2) * ((XY_r - self.c2) @ self.A2.T), dim=-1)).reshape(*XY.shape[:-1])
        weight_factor = weight_factor_1 / (weight_factor_1 + weight_factor_2)
        F = F1 * weight_factor + F2 * (1 - weight_factor)
        return F
    
    def seperatrix(self, X):
        return 1.0 #self.start_height + (self.end_height - self.start_height) * torch.sqrt(X) / 2.0
    
    @staticmethod
    def smooth_threshold_sigmoid(x, x0=3.0, x1=7.0):
        """Smooth step function that transitions from 0 to 1 between x0 and x1."""
        t = (x - x0) / (x1 - x0)
        S = torch.where(
            x <= x0, 0,  # Strictly 0 before x0
            torch.where(x >= x1, 1,  # Strictly 1 after x1
                    3*t**2 - 2*t**3)  # Smooth transition between x0 and x1
        )
        return S
        
    def calc_dzdt_in_safe_region(self, center_x, center_y, side_length, theta, m, p): 
        
        # First, make some of the parameters stay constant over time
        dcenter_xdt = torch.zeros_like(m)
        dcenter_ydt = torch.zeros_like(dcenter_xdt)
        dmdt = torch.zeros_like(dcenter_xdt)
        dthetadt = torch.zeros_like(dcenter_xdt)
        
        # Normalize the side_length to a [0, 1.0 * scaling_factor] interval
        l = side_length / (max(self.resolution))
        l = l * self.scaling_factor #(l - 0.1) * self.scaling_factor
        
        # Define whether the 'x' variable is p or side_length
        name_val_pair = {"x": ("p", p, self.equilibrium_p), 
                    "y": ("l", l, self.scaling_factor * self.equilibrium_side_length / (max(self.resolution)))}
        
        # Assign the just mentioned values
        x = name_val_pair["x"][-2]
        y = name_val_pair["y"][-2]
        
        # We want two behaviors: limit cycles and convergence to an equilibrium. The sigma determines in which region 
        # we are and in which region we are not.
        sigma = self.smooth_threshold_sigmoid((y - self.seperatrix(x)), x0=-0.05, x1=0.05)
        multiplier = 10.0
        sigma_1 = 2.0 * torch.exp(-multiplier * (y - self.seperatrix(x))) #/ (torch.exp(multiplier * -(y-self.seperatrix(x))) + torch.exp(multiplier * (y-self.seperatrix(x))))
        sigma_2 = 2.0 * torch.exp(multiplier * (y - self.seperatrix(x))) #/ (torch.exp(multiplier * -(y-self.seperatrix(x))) + torch.exp(multiplier * (y-self.seperatrix(x))))
        
        # First, define the limit cycle dynamics
        x_clone, y_clone = x.clone(), y.clone()
        x_clone.requires_grad_(True)
        y_clone.requires_grad_(True)
        F_val = self.F(x_clone, y_clone)
        grad_F = torch.autograd.grad(F_val.sum(), (x_clone, y_clone))
        dxdt_1 = grad_F[1] - grad_F[0] * (self.F(x, y) - self.level_set_value)
        dydt_1 = -grad_F[0] - grad_F[1] * (self.F(x, y) - self.level_set_value)
        dxdt_1 = 5.0 * dxdt_1
        dydt_1 = 5.0 * dydt_1
        # dxdt_1 = y - x * (self.F(x, y) - self.level_set_value)
        # dydt_1 = -x - y * (self.F(x, y) - self.level_set_value)
        
        # dxdt_1 = dxdt_1 + sigma_1 * torch.abs(dxdt_1)
        dydt_1 = dydt_1 + sigma_1 * torch.abs(dydt_1)  
        
        # dxdt_1 = dxdt_1 - sigma_1 * dydt_1
        # dydt_1 = dydt_1 + sigma_1 * dxdt_1  
        
        # Then the equilibrium dynamics
        equilibrium_x = name_val_pair["x"][-1]
        equilibrium_y = name_val_pair["y"][-1]
        # dxdt_2 = -(x - equilibrium_x) + 0.25 * torch.sin(y - equilibrium_y)
        # dydt_2 = -(y - equilibrium_y) + 0.25 * torch.sin((x - equilibrium_x))
        
        alpha, beta = 0.5, 5.0
        dxdt_2 = -alpha * (x - equilibrium_x) - beta * (y - equilibrium_y) + 0.25 * torch.sin(y - equilibrium_y) 
        dydt_2 = beta * (x - equilibrium_x) - alpha * (y - equilibrium_y) + 0.25 * torch.sin((x - equilibrium_x))
        
        dxdt_2 = dxdt_2 - sigma_2 * torch.abs(dxdt_2)
        dydt_2 = dydt_2 - sigma_2 * torch.abs(dydt_2)
        
        # Get the total dynamics by combining the two via sigma (which determines in which region we are and what dynamics we have)
        
        if self.only_region_1:
            dxdt = dxdt_1
            dydt = dydt_1
        else:
            dxdt = sigma * dxdt_1 + (1 - sigma) * dxdt_2
            dydt = sigma * dydt_1 + (1 - sigma) * dydt_2
        

        
        # We also want to avoid values of l outside the interval [0.0, self.scaling_factor] and values of p outside the interval [0.0, 4.0]. We use the following values for this
        # dxdt = (x / (torch.abs(x) + 0.001)) * dxdt
        # dydt = (y / (torch.abs(y) + 0.001)) * dydt

        # Assigning the correct values to the correct thing
        for dt, state in zip([dxdt, dydt], ["x","y"]):
            if name_val_pair[state][0] == "p":
                dpdt = dt
            elif name_val_pair[state][0] == "l":
                dside_lengthdt = dt
                dside_lengthdt = dside_lengthdt * max(self.resolution) / self.scaling_factor
        
        # Finally, return the full order dynamics
        return torch.stack([dcenter_xdt, dcenter_ydt, dside_lengthdt, dthetadt, dmdt, dpdt], dim=0)
        
#########################################################################
### Defining some utils for creating and saving latent codes and gifs ###
#########################################################################


def create_time_series_gif(image_array, save_dir, time_series_id):
    r"""Saving a gif of the image time series.

    Args:
        image_array (numpy array of size (N, resolution_x, resolution_y): numpy array containing ima single image time series.

        save_dir (Path): path to the directory where we save the gif.

        time_series_id (int): an integer indicating the id of the track / time series
    """

    # Create a figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Add a colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    p = ax.imshow(image_array[0].squeeze(), vmin=0, vmax=image_array.max(), origin='lower')
    cb = fig.colorbar(p, cax=cax)
    tx = ax.set_title("Ground Truth at time point {}".format(0))
    
    def animate(i):
        p.set_data(image_array[i].squeeze())
        p.set_clim(0, image_array.max())
        tx.set_text("Ground Truth at time point {}".format(i))
        return p,

    gif = FuncAnimation(fig, animate, blit=True, repeat=True, frames=image_array.shape[0], interval=1)
    gif.save(os.path.join(save_dir, "TimeseriesGif_{}.gif".format(time_series_id)), dpi=150,
             writer=PillowWriter(fps=5))
    ax.clear()

    # Close the figures
    plt.close('all')


def save_squares_latents_and_gifs(name_squares_dataset: str, squares: np.ndarray, z_t: np.ndarray):
    r"""Saving a squares dataset.

    Args:
        name_squares_dataset (str): a name for the time series

        squares (numpy array of size (num_time_points, N, resolution_x, resolution_y)): the square image time series.

        z_t (numpy array of size (num_time_points, N, 4)): time series of the latent variables z = [center_x, center_y, side_length, theta].
    """

    # Get the directory where we save the square images as .tif file
    save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", name_squares_dataset)

    # Create some directories if they do not exist
    if not os.path.isdir(os.path.join(save_dir, "latents")):
        os.makedirs(os.path.join(save_dir, "latents"))
    if not os.path.isdir(os.path.join(save_dir, "gifs")):
        os.makedirs(os.path.join(save_dir, "gifs"))

    # For every time series
    for i in range(squares.shape[1]):

        # We save each time point separately as a .tif file
        for j in range(squares.shape[0]):

            # Save the image as a .tif file
            img_obj = Image.fromarray(squares[j, i, ...])
            time_label = "00" + str(j) if j < 10 else "0" + str(j)
            save_filename = "track{}_t{}.tif".format(i, time_label)
            dirname = os.path.join(save_dir, "images", "Track{}".format(i))
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            img_obj.save(os.path.join(dirname, save_filename))

        # Save the latent vectors defining the images
        np.save(os.path.join(save_dir, "latents", "Track{}_latent.npy".format(i)), z_t[:, i, :])

        # Finally, save some gifs of the time series
        create_time_series_gif(squares[:, i, ...], os.path.join(save_dir, "gifs"), i)



if __name__ == '__main__':
    
    ##################################
    ### Now defining a time series ###
    ##################################

    # Define the number of time series and the number of time points
    num_time_series = 50
    num_time_points = 50

    # Define the resolution in each direction
    resolution_x = 64
    resolution_y = 64

    # Define the time between two time steps
    nabla_t = 0.1
    
    # Put in the values necessary for generating the dataset
    force_field_magnitude = 20.0 * min(resolution_x, resolution_y) / 12
    alpha = 1.0
    beta = 0.1
    gamma = 0.05
    delta = 1.0
    epsilon = 0.05
    #name_dataset = "RegulatedGrowth"
    name_dataset = "LimitCyclesAndEquilibria"
    
    # Get the object that can be used to create time series
    # time_series_creator = LpTimeSeriesRegulatedGrowth(shift=int(min(resolution_x, resolution_y) / 8), 
    #                                                   force_field_magnitude=force_field_magnitude, 
    #                                                   resolution=(resolution_x, resolution_y), 
    #                                                   smooth_function="molifier", 
    #                                                   alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon)
    time_series_creator = LpTimeSeriesLimitCyclesAndEquilibria(shift=int(min(resolution_x, resolution_y) / 16), 
                                                               force_field_magnitude=2.0, 
                                                               resolution=(resolution_x, resolution_y), 
                                                               smooth_function="molifier")

    # Get some random initial conditions
    center0_x = resolution_x / 2 + np.zeros(num_time_series) 
    center0_y = resolution_y / 2 + np.zeros(num_time_series) 
    side_length0 = min(resolution_x, resolution_y) / 8 + min(resolution_x, resolution_y) / 4 * np.random.rand(
        num_time_series)
    m0 = np.ones(num_time_series)
    p0 = 0.5 + np.random.rand(num_time_series) * 1.5
    theta0 = np.zeros(num_time_series) 
    noise_stds = [0.05, 0.1]
    
    # Generate random square time series
    z_t, shapes, shapes_noisy_list = time_series_creator.calculate_image_path(center0_x, center0_y, side_length0, theta0, m0, p0, num_time_points, nabla_t, noise_stds=noise_stds)

    # Save the squares and gifs
    save_squares_latents_and_gifs(name_dataset, shapes.numpy(), z_t.numpy())
    for noise_std, shapes_noisy in zip(noise_stds, shapes_noisy_list):
        save_squares_latents_and_gifs(name_dataset + "_noisy_{}".format(noise_std), shapes_noisy.numpy(), z_t.numpy())
