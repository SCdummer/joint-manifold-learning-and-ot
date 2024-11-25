import re
from pathlib import Path

import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader

from ml.util import register


@register('dataset', 'HeLa')
class HeLaCells(VisionDataset):
    IMG_SIZE = (64, 64)

    NORMALIZE = T.Normalize(mean=[0.02108], std=[0.07335])
    DE_NORMALIZE = T.Normalize(mean=[-0.02108 / 0.07335], std=[1 / 0.07335])

    DEFAULT_TRANSFORM = T.Compose([
        T.ToTensor(),
    ])

    BASE_FOLDER = 'Fluo-N2DL-HeLa'
    TRACKS_FOLDER = '01_processed'

    def __init__(self, root, split='train', seed=None, test_size=0.2, transform=None, target_transform=None):
        root = Path(root, self.BASE_FOLDER, self.TRACKS_FOLDER)
        super(HeLaCells, self).__init__(root)

        # get all the tracks, these are folders in the root_dir
        tracks = [d for d in root.iterdir() if d.is_dir() and re.match(r'Track\d+', d.name)]
        print(f'Found {len(tracks)} tracks')

        if seed is not None:
            train_tracks, test_tracks = train_test_split(tracks, random_state=seed, test_size=test_size)
            tracks = train_tracks if split == 'train' else test_tracks

        all_images_per_track = {
            track: sorted(list(track.glob('*.png')))
            for track in tracks
        }

        # we need to sample pairs of images that are consecutive in time
        self.image_pairs = []
        for track, images in all_images_per_track.items():
            for i in range(len(images) - 1):
                self.image_pairs.append((images[i], images[i + 1]))

        if transform is None:
            transform = self.DEFAULT_TRANSFORM

        if target_transform is None:
            target_transform = self.DEFAULT_TRANSFORM

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        x0_path, x1_path = self.image_pairs[idx]
        x0, x1 = pil_loader(x0_path), pil_loader(x1_path)

        # add one channel to the images as they are grayscale at the end
        x0 = x0.convert('L')
        x1 = x1.convert('L')

        x0 = self.transform(x0)
        x1 = self.transform(x1)

        return x0, x1


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    _ds = HeLaCells(
        Path('..', '..', 'data'), seed=42,
        split='val', test_size=0.2
    )
    print(repr(_ds))

    for _i in range(10):
        _x0, _x1 = _ds[_i]
        print(_x0.shape, _x1.shape)
        _, _axs = plt.subplots(1, 2, figsize=(10, 5))
        _axs[0].imshow(HeLaCells.DE_NORMALIZE(_x0).squeeze().numpy(), cmap='gray')
        _axs[1].imshow(HeLaCells.DE_NORMALIZE(_x1).squeeze().numpy(), cmap='gray')
        plt.show()
