import re
from pathlib import Path

import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset


class BaseDataset(VisionDataset):
    DEFAULT_TRANSFORM = T.Compose([
        T.ToTensor(),
    ])

    EXTENSION = 'tif'

    def __init__(self, root, split='train', seed=None, test_size=0.2, transform=None, full_time_series=False):
        super(BaseDataset, self).__init__(root=Path(root))

        if not self.root.exists():
            # extract the zip file which has the same name as the root folder + '.zip'
            import zipfile
            zip_file = self.root.with_suffix('.zip')
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(self.root.parent)

        # get all the tracks, these are folders in the root_dir
        tracks = [d for d in self.root.iterdir() if d.is_dir() and re.match(r'Track\d+', d.name)]
        print(f'Found {len(tracks)} tracks')

        # get the extension of the tracks and make sure it is a unique extension
        extensions = [suffix for d in tracks for suffix in [path.suffix for path in list(d.glob(f'*.*'))]]
        if not len(set(extensions)) == 1:
            raise ValueError("Dataset at path {} does not have a unique extension for the images. We found: {}".format(self.root, set(extensions)))
        else:
            self.EXTENSION = extensions[0][1:]

        # print the minimum and maximum number of images in a track
        print(
            'Min number of images:', min(
                len(list(d.glob(f'*.{self.EXTENSION}'))) for d in tracks
            )
        )
        print(
            'Max number of images:', max(
                len(list(d.glob(f'*.{self.EXTENSION}'))) for d in tracks
            )
        )

        train_tracks_HeLa = ['Track151', 'Track12', 'Track137', 'Track51', 'Track363', 'Track185', 'Track199', 'Track86', 'Track190', 'Track16', 'Track197', 'Track305', 'Track353', 'Track94', 'Track1', 'Track326', 'Track315', 'Track133', 'Track169', 'Track366', 'Track102', 'Track55', 'Track225', 'Track112', 'Track371', 'Track25', 'Track380', 'Track161', 'Track24', 'Track20', 'Track98', 'Track204', 'Track141', 'Track63', 'Track186', 'Track30', 'Track41']
        train_tracks_Gaussian = ['Track2', 'Track12', 'Track42', 'Track16', 'Track11', 'Track14', 'Track46', 'Track6', 'Track7', 'Track22', 'Track17', 'Track23', 'Track30', 'Track4', 'Track37', 'Track0', 'Track49', 'Track33', 'Track39', 'Track13', 'Track35', 'Track19', 'Track41', 'Track1', 'Track28', 'Track10', 'Track48', 'Track40', 'Track3', 'Track45', 'Track18', 'Track29', 'Track25', 'Track9', 'Track27', 'Track15', 'Track47', 'Track21', 'Track34', 'Track43']
        test_tracks_HeLa = ['Track288', 'Track50', 'Track284', 'Track76', 'Track257', 'Track372', 'Track177', 'Track200', 'Track125', 'Track270']
        test_tracks_Gaussian = ['Track20', 'Track44', 'Track36', 'Track5', 'Track24', 'Track8', 'Track32', 'Track31', 'Track38', 'Track26']
        tracks_names = [track.name for track in tracks]
        
        if seed is not None:
            
            if set(train_tracks_HeLa).issubset(tracks_names):
                train_tracks = [Path(self.root, track) for track in train_tracks_HeLa]
                test_tracks = [Path(self.root, track) for track in test_tracks_HeLa]
            else:
                train_tracks = [Path(self.root, track) for track in train_tracks_Gaussian]
                test_tracks = [Path(self.root, track) for track in test_tracks_Gaussian]
                test_tracks = train_tracks+test_tracks
                test_tracks.sort()
            
            #train_tracks, test_tracks = train_test_split(tracks, random_state=seed, test_size=test_size)
            
            tracks = train_tracks if split == 'train' else test_tracks

        self.all_images_per_track = {
            track: list(sorted(list(track.glob(f'*.{self.EXTENSION}'))))
            for track in tracks
        }
        self.all_images_per_track = {
            track: (
                images,
                int(re.match(rf'track(\d+)_t(\d+).{self.EXTENSION}', images[0].name).group(2)),
                int(re.match(rf'track(\d+)_t(\d+).{self.EXTENSION}', images[-1].name).group(2))
            )
            for track, images in self.all_images_per_track.items()
        }
        self.track_weights = {
            track: 1 / len(images) for track, (images, _, _) in self.all_images_per_track.items()
        }

        # Get the lowest starting time and largest end time
        self.start_time = min([start_time for _, start_time, _ in self.all_images_per_track.values()])
        self.end_time = max([end_time for _, _, end_time in self.all_images_per_track.values()])

        # we need to sample pairs of images that are n consecutive in time
        # thereofore, if n=0, it is only the image itself, n=1 is the image and the next one, etc.
        self.image_tracks = []
        self.sample_weights = []
        for track, (images, start_time, end_time) in self.all_images_per_track.items():
            for i in range(len(images) - 1):
                self.image_tracks.append((images[i], images[i + 1]))
                self.sample_weights.append(self.track_weights[track])

        if transform is None:
            transform = self.DEFAULT_TRANSFORM

        self.transform = transform
        self.split = split
        self.full_time_series = full_time_series

    def get_sampling_weights(self):
        return self.sample_weights

    def get_full_track(self, track_idx):
        if isinstance(track_idx, str):
            track_images = self.all_images_per_track[track_idx][0]
        elif isinstance(track_idx, int):
            track_images = self.all_images_per_track[list(self.all_images_per_track.keys())[track_idx]][0]
        else:
            raise ValueError('track_idx must be either a string or an integer')
        return list(
            map(lambda x: self.DEFAULT_TRANSFORM(np.array(Image.open(x))[..., None]), track_images)
        )

    def get_tracks_selected(self):
        return list(self.all_images_per_track.keys())

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class HeLaCellsSuccessive(BaseDataset):

    def __init__(
            self, root, n_successive=0, subsampling=1, full_time_series=False,
            split='train', seed=None, test_size=0.2, transform=None
    ):
        super(HeLaCellsSuccessive, self).__init__(root, split, seed, test_size, transform, full_time_series)
        self.n_successive = n_successive
        self.subsampling = subsampling

        self.image_tracks = {}
        self.sample_weights = []
        mult = self.subsampling
        for track_id, (track, (images, start_time, end_time)) in enumerate(self.all_images_per_track.items()):
            self.image_tracks[track_id] = []
            for i in range(0, len(images) - mult * n_successive, mult):
                if i + mult * n_successive + 1 > len(images):
                    break

                self.image_tracks[track_id].append(
                    (
                        images[i:i + mult * n_successive + 1:mult],
                        ((start_time - self.start_time + i) / self.end_time,
                         (start_time - self.start_time + i + n_successive * mult) / self.end_time)
                    )
                )
                self.sample_weights.append(self.track_weights[track])

        self.flat_image_tracks = sum(self.image_tracks.values(), [])

    @staticmethod
    def get_collate_fn():
        # stack the images along the batch dimension using concatenation
        def collate_fn(batch):
            # batch is a list of tuples (x, t) where x is a list of images [(c, h, w) * n] and t is the time (n,)
            # first chunk should be the images[0] of each track, second chunk should be images[1] of each track etc.
            x, t = zip(*batch)  # x is a list of [(c, h, w) * n] and t is a list of (n,)
            x, t = list(zip(*x)), list(zip(*t))
            x, t = [torch.stack(x_i, dim=0) for x_i in x], [torch.stack(t_i, dim=0) for t_i in t]
            x, t = torch.concat(x, dim=0), torch.concat(t, dim=0)
            return x, t

        return collate_fn

    def get_sampling_weights(self):
        return self.sample_weights

    def __len__(self):
        if self.full_time_series:
            # if we are getting full time series then we just get the tracks in train / val set and return all images
            # with no subsampling
            return len(self.all_images_per_track)
        else:
            # else we return the number of image tracks after subsampling and taking into account the number of
            # successive images
            return len(self.flat_image_tracks)

    def __getitem__(self, idx):
        if self.full_time_series:
            xs, start_time, end_time = self.all_images_per_track[list(self.all_images_per_track.keys())[idx]]
            xs = xs[::self.subsampling]
            t = torch.linspace(start_time / end_time, 1, steps=len(xs), dtype=torch.float32)
        else:
            xs, times = self.flat_image_tracks[idx]
            t = torch.linspace(times[0], times[1], steps=self.n_successive + 1, dtype=torch.float32)
        xs = [np.array(Image.open(x))[..., None] for x in xs]
        xs = [self.transform(x) for x in xs]
        return xs, t


class SquaresMovingHorizontallySuccessive(HeLaCellsSuccessive):
    BASE_FOLDER = 'Squares_horizontally_moving'
    TRACKS_FOLDER = 'images'


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    _n_successive = 1
    _ds = HeLaCellsSuccessive(
        Path('..', '..', 'data', 'Fluo-N2DL-HeLa', '01_processed'), n_successive=_n_successive, subsampling=5,
        split='test', test_size=0.2, seed=42, full_time_series=True
    )

    # print(_ds.get_tracks_selected())
    print([track.name for track in _ds.get_tracks_selected()])
    print(len(_ds.get_full_track(0)))
    print(_ds.get_full_track(0)[0].shape)
