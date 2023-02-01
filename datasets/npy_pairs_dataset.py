import os
import numpy as np
from torch.utils.data import Dataset


class NPYPairsDataset(Dataset):
    def __init__(self, root: str, num_src_channels: int, transforms=None):
        self.extension=".npy"
        self.transforms = transforms
        self.root = root
        self.pathes = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        assert len(self.pathes) != 0, f"No files found in provided directory, {root}."
        self.pathes = [path for path in self.pathes if path.endswith(self.extension)]
        assert len(self.pathes) != 0, \
            f"No files with the '{self.extension}' extension found in provided directory, {root}."

        self.num_channels = num_src_channels
        assert num_src_channels > 0 and isinstance(num_src_channels, int), \
            f"Parameter num_channels ({num_src_channels}) should be positive number."

    def __getitem__(self, idx):
        item = np.load(self.pathes[idx])
        if self.transforms is not None:
            return self.transforms(item)
        return item

    def __len__(self):
        return len(self.pathes)
