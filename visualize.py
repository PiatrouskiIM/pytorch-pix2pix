import argparse
import os
import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import NPYPairsDataset
from models.models import GeneratorUNet, Discriminator


@torch.no_grad()
def visualize(data, generator, device="cpu"):
    def postprocess(tensor):
        return ((np.transpose(tensor.detach().cpu().numpy(), (0, 2, 3, 1)) + 1) / 2.0 * 255.0).astype(np.uint8)

        # grid = vutils.make_grid(tensor)
        # return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    def split_in_images(arr, n=3):
        return np.stack([arr[..., i * n:(i + 1) * n] for i in range(arr.shape[-1] // n)], axis=0)

    generator.to(device)
    generator.eval()

    batch_src_cpu, batch_tgt_cpu = data
    batch_size = batch_src_cpu.size(0)

    targets = postprocess(batch_tgt_cpu)

    cells = []
    for image_i in range(batch_size):
        src = batch_src_cpu[image_i, None].to(device)

        generated_images = split_in_images(postprocess(generator(src)))
        print(generated_images.shape)
        print(generated_images[0,0].shape)
        # print(reduce_color_pallete(generated_images[0,0]).shape)
        # generated_images = np.stack([reduce_color_pallete(generated_images[0,0]),reduce_color_pallete(generated_images[1,0])])[:,None]
        print(generated_images.shape)
        visualization = np.concatenate([generated_images,
                                        split_in_images(targets[image_i, None])], axis=-3)
        visualization = np.concatenate(visualization, axis=-3)
        visualization = np.concatenate((postprocess(src), visualization), axis=-3)
        cells.append(visualization[0])

    all_vis = np.concatenate(cells, axis=-2)
    cv2.imshow("A", all_vis)
    cv2.waitKey()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="/home/ivan/Experiments/JAN27-FACE-PIXELED-COLORS", help="")
    parser.add_argument("--data", type=str, default="/home/ivan/Datasets/JAN30-NPY-DATASET/test",
                        help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
    parser.add_argument("--height", type=int, default=256, help="size of image height")
    parser.add_argument("--width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--checkpoint", type=str, default="0001", help="name of the dataset")
    parser.add_argument("--device", type=str, default="cpu", help="name of the dataset")
    args = parser.parse_args()

    device = args.device
    size = (args.height, args.width)
    lr, betas = args.lr, (args.b1, args.b2)
    imgs_channels = args.channels
    batch_size = args.batch_size
    dataset_path = args.data
    num_epochs = args.epochs
    num_workers = args.workers
    save_folder = args.folder
    prefix = str(args.checkpoint).zfill(4)

    loader_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize(mean=[.5 for _ in range(9)], std=[.5 for _ in range(9)])
    ])
    dataset = NPYPairsDataset(dataset_path, num_src_channels=imgs_channels, transforms=loader_transforms)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1)

    modules = {"G": GeneratorUNet(), "D": Discriminator()}
    if len(os.listdir(save_folder)):
        for net_type in modules:
            path = os.path.join(save_folder, f'{prefix}-{net_type}.ckpt')
            if os.path.isfile(path):
                print("LOADED!")
                modules[net_type].load_state_dict(torch.load(path))
    data = next(dataloader.__iter__())
    data = (data[:, :3], data[:, 3:])

    visualize(data=data, generator=modules["G"], device=device)
