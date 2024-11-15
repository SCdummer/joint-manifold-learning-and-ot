import os
import re

import torch
import numpy as np
from PIL import Image


class CellData(torch.utils.data.Dataset):
    def __init__(self, data_dir, time_step=10, subsample=10):
        super(CellData, self).__init__()

        # Save the inputs
        self.data_dir = data_dir
        self.time_step = time_step
        self.subsample = subsample

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

        # Finally, process the key values

        # Get the tracks that you have
        self.track_ids = list(self.data_dict.keys())


    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        return self.data_dict[self.track_ids[idx]]

