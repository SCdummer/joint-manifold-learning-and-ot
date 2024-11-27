import re
from pathlib import Path

import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader

from ml.util import register


@register('dataset', 'HeLaPair')
class HeLaCells(VisionDataset):
    IMG_SIZE = (64, 64)

    DEFAULT_TRANSFORM = T.Compose([
        T.ToTensor(),
    ])

    BASE_FOLDER = 'Fluo-N2DL-HeLa'
    TRACKS_FOLDER = '01_processed'

    EXTENSION = 'tif' if TRACKS_FOLDER == '01_processed_c' else 'png'
    if TRACKS_FOLDER == '01_processed_c':
        NORMALIZE = T.Normalize(mean=[0.039523557904693814], std=[0.09271456955207719])
        DE_NORMALIZE = T.Normalize(mean=[-0.039523557904693814 / 0.09271456955207719], std=[1 / 0.09271456955207719])
    else:
        NORMALIZE = T.Normalize(mean=[0.02108], std=[0.07335])
        DE_NORMALIZE = T.Normalize(mean=[-0.02108 / 0.07335], std=[1 / 0.07335])

    def __init__(self, root, split='train', seed=None, test_size=0.2, transform=None):
        root = Path(root, self.BASE_FOLDER, self.TRACKS_FOLDER)
        super(HeLaCells, self).__init__(root)

        # get all the tracks, these are folders in the root_dir
        tracks = [d for d in root.iterdir() if d.is_dir() and re.match(r'Track\d+', d.name)]
        print(f'Found {len(tracks)} tracks')
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

        if seed is not None:
            train_tracks, test_tracks = train_test_split(tracks, random_state=seed, test_size=test_size)
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
            self, root, n_successive=1, subsampling=1, split='train', seed=None, test_size=0.2, transform=None
    ):
        super(HeLaCellsSuccessive, self).__init__(root, split, seed, test_size, transform)
        self.n_successive = n_successive
        self.subsampling = subsampling

        self.image_tracks = {}
        mult = self.subsampling
        for track_id, (track, (images, start_time, end_time)) in enumerate(self.all_images_per_track.items()):
            self.image_tracks[track_id] = []
            for i in range(len(images) - n_successive):
                if i + mult * n_successive + 1 > len(images):
                    break

                self.image_tracks[track_id].append(
                    (
                        images[i:i + mult * n_successive + 1],
                        ((start_time + i) / end_time, (start_time + i + n_successive * mult) / end_time)
                    )
                )
        self.flat_image_tracks = sum(self.image_tracks.values(), [])

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
        return len(self.image_tracks) if self.n_successive > 0 else len(self.flat_image_tracks)

    def __getitem__(self, idx):
        if self.n_successive != 0:
            successive_tracks = self.image_tracks[idx]
            # same a random successive track index
            rand_successive_frames = torch.randint(0, len(successive_tracks), (1,)).item()
            xs, times = successive_tracks[rand_successive_frames]
        else:
            xs, times = self.flat_image_tracks[idx]
        xs = [np.array(Image.open(x), dtype=np.float32)[..., None] / 255. for x in xs[::self.subsampling]]
        # min_xs, max_xs = np.min(xs), np.max(xs)
        # xs = [(x - min_xs) / (max_xs - min_xs) for x in xs]
        xs = [self.transform(x) for x in xs]

        # stack the images along a new dimension
        t = torch.linspace(times[0], times[1], steps=self.n_successive + 1, dtype=torch.float32)
        return xs, t  # [(1, 64, 64) * n_successive], (n_successive,)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    _n_successive = 0
    _ds = HeLaCellsSuccessive(
        Path('..', '..', 'data'), n_successive=_n_successive, subsampling=1,
        split='train', test_size=0.2, seed=None,
    )

    print(_ds.get_tracks_selected())
    print(_ds.get_full_track(0))

    _dl = iter(
        torch.utils.data.DataLoader(
            _ds, batch_size=64 // (_n_successive + 1),
            # sampler=torch.utils.data.WeightedRandomSampler(
            #     _ds.get_sampling_weights(), 200, replacement=True
            # ),
            collate_fn=_ds.get_collate_fn()
        )
    )

    print(repr(_ds))

    mean, std = 0, 0
    for _i in range(len(_dl)):
        _x, _t = next(_dl)
        mean += torch.mean(_x).item()
        std += torch.std(_x).item()
        continue

        print(_x.shape, _t.shape)

        skipper = _x.size(0) // (_n_successive + 1)

        _x = _x[::skipper]
        _t = _t[::skipper]

        print(_x.shape, _t.shape)

        _, _ax = plt.subplots(1, _n_successive + 1, figsize=(4 * (_n_successive + 1), 4))
        for _i, (_x_i, _t_i) in enumerate(zip(_x, _t)):
            print(f'Min: {torch.min(_x_i[0])}, Max: {torch.max(_x_i[0])}')
            _normed_xi = (_x_i[0] - torch.min(_x_i[0])) / (torch.max(_x_i[0]) - torch.min(_x_i[0]))
            _ax[_i].imshow(_normed_xi, cmap='gray')
            _ax[_i].set_title(f'{_t_i:.2f}')
            _ax[_i].axis('off')
        plt.show()
        plt.close()

    print(mean / len(_dl), std / len(_dl))
