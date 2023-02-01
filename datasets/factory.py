# The dataset consists of pairs of pictures for the pix2pix model or pairs (picture, type)
# for the cyclegan model. This repository focuses on the PictuPix model, so we will be
# interested in image pair datasets. There are several ways to represent images as a pair,
# we will briefly discuss some of them.
#
# - Specifying a pair using file names or folder structure allows you to link pictures of
#  different sizes.
#
# - Specifying a pair of pictures by combining them into one picture by concatenating in
#  width allows you to view pictures using the operating system.
#
# - Specifying a pair by combining them into one array by concatenating the channels of
#  each element of the pair is appropriate when the element of the pair has more than 4
#  channels or for some other reason it is not necessary to work with it
#  in the operating system.
#
# Since any of the presented options solves the problem, and the latter is
# a generalization, we chose to implement only the last two options.

from .npy_pairs_dataset import NPYPairsDataset
from .cat_images_dataset import CatImagesDataset


def get_dataset(root, num_src_channels=3, num_total_channels=6, axis=-1, transforms=None):
    assert axis in [-1, -2], "Assuming 3d input tensors, B x H x W x C, `axis` parameter must be either -2 or -1 " \
                             "for width or channel axis respectively."
    if axis == -1:
        return NPYPairsDataset(root, num_src_channels=num_src_channels, transforms=transforms)
    return CatImagesDataset(root, num_images=num_total_channels//num_src_channels, transforms=transforms)
