import os
import re

import torch
import numpy as np
from PIL import Image


class CellData(torch.utils.data.Dataset):
    def __init__(self, data_dir, time_step=10, dynamic=True, full_time_series=False):
        super(CellData, self).__init__()

        # Save the inputs
        self.data_dir = data_dir
        self.time_step = time_step
        self.dynamic = dynamic
        self.full_time_series = full_time_series

        # Define a dictionary that will contain the data
        self.data_dict = {}

        # Save the maximum value of the images and the minimum value. This is used for normalization purposes.
        self.max_val = -100000000000000
        self.min_val = 100000000000000

        # Load all the data in a dictionary
        for folder in os.listdir(data_dir):

            # Get the track id
            track_id = int(re.search(r'\d+$', folder).group())

            # Initialize the entry corresponding to this id
            self.data_dict[track_id] = []

            # Grab all the images
            img_filenames = sorted(os.listdir(os.path.join(data_dir, folder)))

            # Load the images and add them to the dictionary
            for filename in img_filenames:

                # Get the time
                time = int(filename[-7:-4])

                # Skip this one if it does not satisfy the time step
                if time % time_step == 0:

                    # Load the image and add to the dictionary
                    img = np.array(Image.open(os.path.join(data_dir, folder, filename)))
                    if len(img.shape) == 2:
                        img = img[np.newaxis, :, :]
                    self.data_dict[track_id].append(img)

                    # Update the maximum and minimum values
                    self.max_val = max(self.max_val, img.max())
                    self.min_val = min(self.min_val, img.min())

                else:
                    continue

        # We need a mapping from an index to two consecutive time points as the latter will be sampled. We also have
        # the same but then for the individual images
        idx_to_track_id_time_pair = {}
        idx_to_track_id_image = {}
        idx_counter_dynamic = 0
        idx_counter_static = 0
        for track_id, img_list in self.data_dict.items():

            # Get the number of time points
            num_time_points = len(img_list)

            # For every image pair, do ...
            for pair_idx in range(num_time_points-1):
                idx_to_track_id_time_pair[idx_counter_dynamic] = (track_id, pair_idx, pair_idx+1)
                idx_counter_dynamic += 1

            # For every image, do ...
            for img_idx in range(num_time_points):
                idx_to_track_id_image[idx_counter_static] = (track_id, img_idx)
                idx_counter_static += 1

        # Save the dictionaries as attributes
        self.idx_to_track_id_time_pair = idx_to_track_id_time_pair
        self.idx_to_track_id_image = idx_to_track_id_image

        # Save the number of image pairs and the number of images
        self.num_img_pairs = idx_counter_dynamic - 1
        self.num_imgs = idx_counter_static - 1

        # Get the tracks that you have
        self.track_ids = list(self.data_dict.keys())

        # Save the scaling factor induced by the calculated maximum and minimal values. This scales all images to
        # the interval [0, 1].
        self.scaling_factor = (1.0 / (self.max_val - self.min_val)).astype(np.float32)

    def __len__(self):
        if self.full_time_series:
            return len(self.track_ids)
        else:
            if self.dynamic:
                return self.num_img_pairs
            else:
                return self.num_imgs

    def __getitem__(self, idx):
        if self.full_time_series:
            return (np.stack(self.data_dict[self.track_ids[idx]], axis=0) - self.min_val) * self.scaling_factor
        else:
            if self.dynamic:
                track_id, img_idx_1, img_idx_2 = self.idx_to_track_id_time_pair[idx]
                data = tuple([(self.data_dict[track_id][i] - self.min_val) * self.scaling_factor for i in range(img_idx_1, img_idx_2+1)])
                return (track_id, img_idx_1, img_idx_2, data)
            else:
                track_id, img_idx = self.idx_to_track_id_image[idx]
                return (track_id, img_idx, ( (self.data_dict[track_id][img_idx] - self.min_val) * self.scaling_factor))

