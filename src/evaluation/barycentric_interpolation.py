from src.evaluation.wasserstein_barycenters import convolutional_barycenter_calculation
import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import argparse

def barycenter_based_interpolation(start_imgs, end_imgs, save_dir, num_time_points=10, barycenter_type='OT', max_val=0.7):
    """
    Barycentric interpolation between start_imgs and end_imgs with num_time_points intermediate points.
    The barycenter is of type 'barycenter_type' (options: l2 or OT). 
    We save the images of the barycenter in save_dir/barycenter_type and use a maximum value of max_val when creating the images. 
    """
    
    # First, check the supplied barycenter type
    if not barycenter_type in ['OT', 'l2']:
        raise ValueError("The input 'barycenter_type' is {}, but can only be 'OT' or 'l2'...".format(barycenter_type))

    # Get the size of the images
    h, w = start_imgs.shape

    # Create a grid
    _y, _x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij'
    )

    # Calculate the weights that we want to use
    t_list = list(range(1, num_time_points))

    # Make the start images and end images into torch tensors
    start_imgs = torch.tensor(np.array(start_imgs))
    end_imgs = torch.tensor(np.array(end_imgs))

    # Get the batch needed for the barycenter calculation
    batch = torch.stack([start_imgs, end_imgs], dim=0)

    # Normalize the batch
    if barycenter_type == 'OT':
        batch_normalized = batch[:, None, None, ...] #(batch / torch.sum(batch, dim=(1, 2), keepdim=True))[:, None, None, ...]
    else:
        batch_normalized = batch[:, None, None, ...]

    # Between every start image and end image, calculate barycenters
    if barycenter_type == 'OT':
        barycenter_list = [batch_normalized[0, ...].cpu().detach().numpy().squeeze().astype(np.float32)]
    else:
        barycenter_list = [start_imgs.cpu().detach().numpy().squeeze().astype(np.float32)]
    
    for t in t_list:

        # Calculate the correct type of barycenters
        if barycenter_type == 'OT':

            # Get the correct weight
            weights = torch.tensor([1.0 - t / (num_time_points-1), t  / (num_time_points-1)])[:, None, None, None, None]

            # Calculate the OT barycenters with the above weights
            barycenters = convolutional_barycenter_calculation(batch_normalized, weights=weights, stab_thresh=1e-30, scaling=0.999)
        else:
            barycenters = start_imgs * (1.0 - t / (num_time_points-1)) + end_imgs * t / (num_time_points-1)

        barycenter_list.append(barycenters.squeeze().detach().cpu().numpy())
    
    barycenters = np.stack(barycenter_list, axis=0)
    save_dir = os.path.join(save_dir, barycenter_type)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for i in range(barycenters.shape[0]):
        fig, ax = plt.subplots()
        plt.imshow(barycenters[i, ...].squeeze().astype(np.float32) / max_val, vmin=0, vmax=1.0)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(save_dir, 'barycenter_' + str(i) + '.png'), bbox_inches='tight',
                    pad_inches=0)
        plt.close('all')

if __name__ == "__main__":
    
    # For defining the first image in the interpolation, the second image in the interpolation, the directory where we save everything, and the maximum value that we can use in plotting
    arg_parser = argparse.ArgumentParser(description="Barycentric interpolation")
    arg_parser.add_argument(
        "--img_1",
        "-f1",
        dest="filename1",
        default="data/Fluo-N2DL-HeLa/01_processed/Track288/track288_t007.tif",
        # required=True,
        help="The file location of the FIRST end point of the image interpolation.",
    )
    arg_parser.add_argument(
        "--img_2",
        "-f2",
        dest="filename2",
        default="data/Fluo-N2DL-HeLa/01_processed/Track288/track288_t014.tif",
        # required=True,
        help="The file location of the SECOND end point of the image interpolation.",
    )
    arg_parser.add_argument(
        "--save_dir",
        "-s",
        dest="save_dir",
        default="data/Fluo-N2DL-HeLa",
        # required=True,
        help="The location where we save the interpolation. The interpolations are saved in save_dir/OT and save_dir/l2.",
    )
    arg_parser.add_argument(
        "--max_val",
        "-m",
        dest="max_val",
        type=float,
        default=0.7,
        # required=True,
        help="The maximum value that is used in plotting the barycentric interpolations. " 
        "In particular, the values in the image are divided by this maximum value and then vmin=0 and vmax=1 in the matplotlib.pyplot.imshow() function.",
    )
    args = arg_parser.parse_args()
    
    # Load the images and define the save directory
    img1 = np.array(Image.open(args.filename1))
    img2 = np.array(Image.open(args.filename2))
    save_dir = args.save_dir
    max_val = args.max_val

    # Make them into torch tensors
    img1 = torch.tensor(img1)
    img2 = torch.tensor(img2)

    # Calculate the barycenter based interpolations
    barycenter_based_interpolation(img1, img2, save_dir, num_time_points=8, barycenter_type='OT', max_val=max_val)
    barycenter_based_interpolation(img1, img2, save_dir, num_time_points=8, barycenter_type='l2', max_val=max_val)