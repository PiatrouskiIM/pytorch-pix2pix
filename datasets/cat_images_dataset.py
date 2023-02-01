import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class CatImagesDataset(Dataset):
    def __init__(self, root, num_images, transforms=None):
        self.transforms = transforms
        self.root = root
        self.pathes = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        assert len(self.pathes) != 0, f"No files found in provided directory, {root}."
        self.pathes = [path for path in self.pathes]

        self.num_images = num_images
        assert num_images > 0 and isinstance(num_images, int), \
            f"Parameter num_channels ({num_images}) should be positive number."

    def __getitem__(self, idx):
        item = cv2.imread(self.pathes[idx], cv2.IMREAD_UNCHANGED)
        items = np.split(item, indices_or_sections=self.num_images, axis=-2)
        item = np.concatenate(items, axis=-1)
        if self.transforms is not None:
            return self.transforms(item)
        return item

    def __len__(self):
        return len(self.pathes)
