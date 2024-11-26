import re
from pathlib import Path

import torch
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader

from ml.util import register


@register('dataset', 'HeLaPair')
class HeLaCells(VisionDataset):
    IMG_SIZE = (64, 64)

    NORMALIZE = T.Normalize(mean=[0.02108], std=[0.07335])
    DE_NORMALIZE = T.Normalize(mean=[-0.02108 / 0.07335], std=[1 / 0.07335])

    DEFAULT_TRANSFORM = T.Compose([
        T.ToTensor(),
    ])

    BASE_FOLDER = 'Fluo-N2DL-HeLa'
    TRACKS_FOLDER = '01_processed'

    def __init__(self, root, split='train', seed=None, test_size=0.2, transform=None):
        root = Path(root, self.BASE_FOLDER, self.TRACKS_FOLDER)
        super(HeLaCells, self).__init__(root)

        # get all the tracks, these are folders in the root_dir
        tracks = [d for d in root.iterdir() if d.is_dir() and re.match(r'Track\d+', d.name)]
        print(f'Found {len(tracks)} tracks')

        if seed is not None:
            train_tracks, test_tracks = train_test_split(tracks, random_state=seed, test_size=test_size)
            tracks = train_tracks if split == 'train' else test_tracks

        self.all_images_per_track = {
            track: sorted(list(track.glob('*.png')))
            for track in tracks
        }
        self.all_images_per_track = {
            track: (
                images,
                int(re.match(r'track(\d+)_t(\d+).png', images[0].name).group(2)),
                int(re.match(r'track(\d+)_t(\d+).png', images[-1].name).group(2))
            )
            for track, images in self.all_images_per_track.items()
        }
        self.track_weights = {
            track: 1 / len(images) for track, (images, _, _) in self.all_images_per_track.items()
        }

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

    def get_sampling_weights(self):
        return self.sample_weights

    def get_full_track(self, track_idx):
        if isinstance(track_idx, str):
            track_images = self.all_images_per_track[track_idx][0]
        elif isinstance(track_idx, int):
            track_images = self.all_images_per_track[list(self.all_images_per_track.keys())[track_idx]][0]
        else:
            raise ValueError('track_idx must be either a string or an integer')
        return list(map(lambda x: pil_loader(x).convert('L'), track_images))

    def get_tracks_selected(self):
        return list(self.all_images_per_track.keys())

    def __len__(self):
        return len(self.image_tracks)

    def __getitem__(self, idx):
        x0_path, x1_path = self.image_tracks[idx]
        x0, x1 = pil_loader(x0_path), pil_loader(x1_path)

        # add one channel to the images as they are grayscale at the end
        x0 = x0.convert('L')
        x1 = x1.convert('L')

        x0 = self.transform(x0)
        x1 = self.transform(x1)

        return x0, x1


@register('dataset', 'HeLa')
class HeLaCellsSuccessive(HeLaCells):

    def __init__(
            self, root, n_successive=1, split='train', seed=None, test_size=0.2, transform=None
    ):
        super(HeLaCellsSuccessive, self).__init__(root, split, seed, test_size, transform)
        self.n_successive = n_successive

        self.image_tracks = {}
        for track_id, (track, (images, start_time, end_time)) in enumerate(self.all_images_per_track.items()):
            self.image_tracks[track_id] = []
            for i in range(len(images) - n_successive):
                self.image_tracks[track_id].append(
                    (
                        images[i:i + n_successive + 1],
                        ((start_time + i) / end_time, (start_time + i + n_successive) / end_time)
                    )
                )

    def get_sampling_weights(self):
        return list(self.track_weights.values())

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

    def __len__(self):
        return len(self.image_tracks)

    def __getitem__(self, idx):
        successive_tracks = self.image_tracks[idx]
        # same a random successive track index
        rand_successive_frames = torch.randint(0, len(successive_tracks), (1,)).item()
        xs, times = successive_tracks[rand_successive_frames]
        xs = [pil_loader(x).convert('L') for x in xs]
        xs = [self.transform(x) for x in xs]

        # stack the images along a new dimension
        t = torch.linspace(times[0], times[1], steps=self.n_successive + 1, dtype=torch.float32)
        return xs, t  # [(1, 64, 64) * n_successive], (n_successive,)


if __name__ == '__main__':
    _n_successive = 3
    _ds = HeLaCellsSuccessive(
        Path('..', '..', 'data'), seed=42,
        split='train', test_size=0.2, n_successive=_n_successive,
    )

    print(_ds.get_tracks_selected())
    print(_ds.get_full_track(0))

    _dl = iter(
        torch.utils.data.DataLoader(
            _ds, batch_size=16 // (_n_successive + 1),
            sampler=torch.utils.data.WeightedRandomSampler(
                _ds.get_sampling_weights(), 200, replacement=True
            ),
            collate_fn=_ds.get_collate_fn()
        )
    )

    print(repr(_ds))

    for _i in range(1):
        _x, _t = next(_dl)
        print(_x.shape, _t.shape)
