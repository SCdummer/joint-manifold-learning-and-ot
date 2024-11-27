import os
import shutil

import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm


def get_bounding_box(mask):
    # Convert the mask to a binary format
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the mask")

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h


def process_tracks(data_dir):

    # Load the man_track.txt
    track_dir = os.path.join(data_dir, "01_GT", "TRA")
    man_track_txt_path = os.path.join(track_dir, "man_track.txt")
    man_track_txt = open(man_track_txt_path, "r")

    # Define the save directory
    save_dir = os.path.join(data_dir, "01_processed")

    # Also define the image directory
    img_dir = os.path.join(data_dir, "01")
    
    # Processing the track txt file
    man_track_dict = {}
    for line in man_track_txt.readlines():
        splitted_line = line.split()
        man_track_dict[splitted_line[0]] = splitted_line[1:]

    # Creating a dictionary that saves the tracks
    track_dict = {}

    # Keeping track of the latest track id and the list of labels that we already dealt with
    seen_labels = []

    # For every image, do ...
    for filename in sorted([f for f in os.listdir(track_dir) if f.endswith(".tif")]):

        # Get the complete path
        file_path = os.path.join(track_dir, filename)

        # Load the .tif image
        img = Image.open(file_path)
        img_array = np.array(img)

        # Get all the unique values inside the image array
        labels = np.unique(img_array)

        # Get the time
        time = int(filename[9:12])

        # For every label, do ...
        for label in labels:

            # Make a string version of the label
            label_str = str(label)

            # If the label is zero, then go to the next label
            if label == 0:
                continue

            # Check whether it has a parent
            if man_track_dict[label_str][-1] == "0":
                has_parent = False
                parent_label = None
            else:
                has_parent = True
                parent_label = man_track_dict[label_str][-1]

            # If at a certain moment, we get a track that has divided for more than one times, we do not add these ones
            # We only go up until the point that there are two cells and not more.
            if has_parent:
                if not  man_track_dict[parent_label][-1] == '0':
                    continue

            # Grab the mask of this label
            mask_img = (img_array == label)

            # Add things to track_dict
            if label not in seen_labels:
                seen_labels.append(label)
                if has_parent:
                    if time not in track_dict[parent_label]:
                        track_dict[parent_label][time] = []
                    track_dict[parent_label][time].append(mask_img)
                else:
                    track_dict[label_str] = {}
                    track_dict[label_str][time] = [mask_img]
            else:
                if has_parent:
                    if time not in track_dict[parent_label]:
                        track_dict[parent_label][time] = []
                    track_dict[parent_label][time].append(mask_img)
                else:
                    if time not in track_dict[label_str]:
                        track_dict[label_str][time] = []
                    track_dict[label_str][time].append(mask_img)

    # Create a dictionary that keeps track of the amount of cells in the PREVIOUS IMAGE
    prev_num_cells = {}

    # Also initialize a list of labels that we should not deal with anymore
    label_skip_list = []

    # Create a dictionary similar to track_dict, but then with the images
    track_img_dict = {}

    # Get the lowest possible value in any of the images
    absolute_minimum = 10000000000000000
    for filename in sorted([f for f in os.listdir(img_dir) if f.endswith(".tif")]):

        # Load the image
        img = Image.open(os.path.join(img_dir, filename))
        img = np.array(img)

        # Update the absolute minimum value
        absolute_minimum = min(absolute_minimum, img.min())

    # For every ground truth image, do ...
    for filename in tqdm(sorted([f for f in os.listdir(img_dir) if f.endswith(".tif")]), desc="Processing images", total=len([f for f in os.listdir(img_dir) if f.endswith(".tif")])):
        
        # Get the time
        time = int(filename[1:4])
        
        # Load the image
        img = Image.open(os.path.join(img_dir, filename))
        img = np.array(img)
        
        # For every track, do ...
        for label in track_dict:

            # Check if the time is related to the label, else skip it
            if not time in track_dict[label]:
                continue

            # Add the label and the time to track_img_dict
            if label not in track_img_dict:
                track_img_dict[label] = {}
                track_img_dict[label]['max'] = -10000000000
                track_img_dict[label]['min'] = 1000000000

            # Get the masks at that time point belonging to the label
            masks = track_dict[label][time]
            
            # Get the number of masks
            num_masks = len(masks)

            # Check whether we should skip the current label from now on or not
            if label not in prev_num_cells:
                prev_num_cells[label] = num_masks
            elif label in label_skip_list:
                continue
            else:
                if prev_num_cells[label] > num_masks:
                    # In this case, a cell disappeared. This might happen because there were two cells, but one of them
                    # divided. Then in the previous code, this dividing cell will have been removed, while the
                    # non-dividing cell remains, leaving only 1 cell. We remove these cases here.
                    label_skip_list.append(label)
                    continue
                else:
                    prev_num_cells[label] = num_masks

            # Check if the number of masks is correct. Else generate images
            if num_masks > 2:
                raise ValueError("Must be some mistake in the code")
            else:

                # Define a list of images and a list of masks
                img_list = []
                mask_list = []

                # For every mask, do ...
                for i in range(num_masks):

                    # Get the mask
                    mask_raw = masks[i]

                    # Get the bounding box
                    x, y, w, h = get_bounding_box(mask_raw)

                    # Set some parameters for getting the bounded image
                    num_additional_pixels_x = 10
                    num_additional_pixels_y = 20
                    y = y - num_additional_pixels_y
                    x = x - num_additional_pixels_x
                    h = h + 2 * num_additional_pixels_y
                    w = w + 2 * num_additional_pixels_x

                    # Get the bounds for the bounding box
                    b_lim_y, u_lim_y = (int(y)), (int(y + h))
                    b_lim_x, u_lim_x = (int(x)), (int(x + w))

                    # Alter the values of u_lim_y and u_lim_x. Also adjust b_lim-y and b_lim_x accordingly
                    if not (u_lim_y <= img.shape[0] - 1):
                        u_lim_y = u_lim_y - img.shape[0] - 1
                        b_lim_y = -(img.shape[0] - b_lim_y)

                    if not (u_lim_x <= img.shape[1] - 1):
                        u_lim_x = u_lim_x - img.u_lim_x[0] - 1
                        b_lim_x = -(img.shape[1] - b_lim_x)

                    # Define the mask as the raw mask
                    mask = mask_raw

                    # Define the background color
                    background_color = absolute_minimum #33036

                    # Get the image and mask in the bounding box
                    img_in_bbox = img[np.r_[b_lim_y:u_lim_y], :]
                    img_in_bbox = img_in_bbox[:, np.r_[b_lim_x:u_lim_x]]
                    mask_bbox = mask[np.r_[b_lim_y:u_lim_y], :]
                    mask_bbox = mask_bbox[:, np.r_[b_lim_x:u_lim_x]]

                    # Pad the image with and track mask until we have a specific size
                    target_shape = (64, 32)
                    img_in_bbox = np.pad(img_in_bbox, ((max(int((target_shape[0] - img_in_bbox.shape[0])/2), 0),
                                                     max(int((target_shape[0] - img_in_bbox.shape[0])/2), 0)),
                                                       (max(int((target_shape[1] - img_in_bbox.shape[1]) / 2), 0),
                                                       max(int((target_shape[1] - img_in_bbox.shape[1]) / 2), 0)))
                                         , 'constant', constant_values=background_color)
                    mask_bbox = np.pad(mask_bbox, ((max(int((target_shape[0] - mask_bbox.shape[0])/2), 0),
                                                     max(int((target_shape[0] - mask_bbox.shape[0])/2), 0)),
                                                   (max(int((target_shape[1] - mask_bbox.shape[1]) / 2), 0),
                                                       max(int((target_shape[1] - mask_bbox.shape[1]) / 2), 0))))

                    # Make sure that outside the mask we smoothly go to a uniform background
                    # kernel_size = 10 #int(num_additional_pixels/4)
                    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    # dilation = cv2.morphologyEx(mask_bbox.astype('uint8'), cv2.MORPH_DILATE, kernel, borderType=cv2.BORDER_REPLICATE)
                    # processed_mask = cv2.blur(255 * dilation.astype(np.float64),(kernel_size, kernel_size))
                    # processed_mask = processed_mask / 255
                    processed_image = background_color * (1.0 - mask_bbox) + mask_bbox * img_in_bbox.astype(np.float64)
                    processed_image = processed_image.astype(img_in_bbox.dtype)

                    # Then resize the image to the correct size
                    img_in_bbox = cv2.resize(processed_image, target_shape[::-1])
                    mask_bbox = cv2.resize(mask_bbox.astype(np.uint8), target_shape[::-1]).astype(bool)

                    # Add the image and the mask to the img_list and mask_list
                    img_list.append(img_in_bbox)
                    mask_list.append(mask_bbox)

                # NOTE: due to the cv2.resize, the first axis is now the y-axis and the x-axis is the second axis

                # Save the masks and image list
                if time not in track_img_dict[label]:
                    track_img_dict[label][time] = {}
                track_img_dict[label][time]['masks'] = mask_list
                track_img_dict[label][time]['imgs'] = img_list
                track_img_dict[label]['max'] = max(track_img_dict[label]['max'], max([img.max() for img in img_list]))
                track_img_dict[label]['min'] = min(track_img_dict[label]['min'], min([img.min() for img in img_list]))

    # # Get the minimum and maximum values among all images
    # absolute_max, absolute_min = -10000000, 1000000000
    # for label in track_img_dict:
    #     max_val, min_val = track_img_dict[label]['max'], track_img_dict[label]['min']
    #     absolute_max = max(absolute_max, max_val)
    #     absolute_min = min(absolute_min, min_val)

    # Now we process the images and save them properly
    use_normalization = True
    for label in track_img_dict:
        max_val, min_val = track_img_dict[label]['max'], track_img_dict[label]['min']
        scaling = 1.0 / (max_val - min_val)
        times = list(track_img_dict[label].keys())
        times.remove('max')
        times.remove('min')
        for time in times:

            # Load the image list
            img_list = track_img_dict[label][time]['imgs']

            if use_normalization:

                # Normalize the images properly
                img_list = [(img - min_val) * scaling for img in img_list]

                # New background color
                background_color = 0

            # Now we need to concatenate all the images
            concat_img = np.concatenate(img_list, axis=1)
            if not concat_img.shape[-1] == 2 * target_shape[-1]:
                concat_img = np.pad(concat_img, ((0,0), (int(target_shape[-1] / 2), int(target_shape[-1] / 2))), 'constant', constant_values=background_color)
                concat_img = cv2.resize(concat_img, (2 * target_shape[1], target_shape[0]))

            # Save the image
            img_obj = Image.fromarray((concat_img * 255).astype(np.uint8))
            time_label = "00"+str(time) if time < 10 else "0"+str(time)
            save_filename = "track{}_t{}.png".format(label, time_label)
            dirname = os.path.join(save_dir, "Track{}".format(label))
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            img_obj.save(os.path.join(dirname, save_filename))

    # Remove the tracks with fewer than a certain number of images
    min_amount_time_points = 30
    for folder in os.listdir(save_dir):

        if len(os.listdir(os.path.join(save_dir, folder))) < min_amount_time_points:
            shutil.rmtree(os.path.join(save_dir, folder))


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Fluo-N2DL-HeLa")
    process_tracks(data_dir)
