import os
import re

import torch
import numpy as np
from PIL import Image


class CellData(torch.utils.data.Dataset):
    def __init__(self, data_dir, time_step=10, dynamic=True):
        super(CellData, self).__init__()

        # Save the inputs
        self.data_dir = data_dir
        self.time_step = time_step
        self.dynamic = dynamic

        # Define a dictionary that will contain the data
        self.data_dict = {}

        # Load all the data in a dictionary
        for folder in os.listdir(data_dir):

            # Get the track id
            track_id = re.search(r'\d+$', folder.split('_')[0])

            # Initialize the entry corresponding to this id
            self.data_dict[track_id] = []

            # Grab all the images
            img_filenames = os.listdir(os.path.join(data_dir, folder))

            # Load the images and add them to the dictionary
            for filename in img_filenames:

                # Get the time
                time = int(filename[-7:-4])

                # Skip this one if it does not satisfy the time step
                if time + 1 == 1 or ((time+1) % time_step == 0):

                    # Load the image and add to the dictionary
                    self.data_dict[track_id].append(np.array(Image.open(os.path.join(data_dir, folder, filename))))

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

    def __len__(self):
        if self.dynamic:
            return self.num_img_pairs
        else:
            return self.num_imgs

    def __getitem__(self, idx):
        if self.dynamic:
            track_id, img_idx_1, img_idx_2 = self.idx_to_track_id_time_pair[idx]
            return (self.data_dict[track_id][img_idx_1], self.data_dict[track_id][img_idx_2])
        else:
            track_id, img_idx = self.idx_to_track_id_image[idx]
            return self.data_dict[track_id][img_idx]

