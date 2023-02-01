import os
import torch
import pandas as pd
from tqdm import tqdm

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets import get_dataset
from utils.weight_initializer import WeightInitializer

from models.models import GeneratorUNet, Discriminator
from ops.pix2pix_objective import Pix2pixObjective


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def train_pix2pix_epoch(dataloader, coach, modules, optimizers, device="cpu"):
    epoch_stats = []
    for i_batch, data in enumerate(dataloader):
        data = (data[:, :3], data[:, 3:])
        batch_stats = {}

        optimizers["G"].zero_grad()
        optimizers["D"].zero_grad()
        loss, stats = coach.practice_discrimination(modules["G"], modules["D"], data, device)
        loss.backward()
        optimizers["D"].step()
        batch_stats["D"] = stats

        optimizers["G"].zero_grad()
        optimizers["D"].zero_grad()
        loss, stats = coach.practice_generation(modules["G"], modules["D"], data, device)
        loss.backward()
        optimizers["G"].step()
        batch_stats["G"] = stats

        epoch_stats.append(batch_stats)
    return epoch_stats


def train(save_folder, dataloader, modules, num_epochs, lr, betas, save_interval=1, device="cpu", **kwargs):
    os.makedirs(save_folder, exist_ok=True)

    def _save_checkpoint(prefix):
        for kind in modules:
            saving_path = os.path.join(save_folder, f'{prefix}-{kind}.ckpt')
            torch.save(modules[kind].state_dict(), saving_path)

    def _save_stats():
        records = []
        for epoch_i, epoch_stats in enumerate(stats):
            for batch_i, batch_stats in enumerate(epoch_stats):
                record = {"epoch": epoch_i, "batch": batch_i}
                for kind in batch_stats:
                    for stat in batch_stats[kind]:
                        joint_key = f"{kind.lower()}_{stat.lower()}_loss"
                        record[joint_key] = batch_stats[kind][stat]
                records.append(record)
        pd.DataFrame.from_records(records).to_csv(os.path.join(save_folder, "statistics.csv"))

    coach = Pix2pixObjective(**kwargs)
    for kind in modules:
        modules[kind].to(device)
    optimizers = {kind: torch.optim.Adam(modules[kind].parameters(), lr=lr, betas=betas) for kind in modules}

    stats = []
    for i_epoch in tqdm(range(num_epochs)):
        epoch_stats = train_pix2pix_epoch(dataloader, coach, modules, optimizers, device)
        stats.append(epoch_stats)

        if i_epoch % save_interval == 0:
            _save_checkpoint(prefix=str(i_epoch).zfill(4))
            _save_stats()

    _save_checkpoint("last")
    _save_stats()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60, help="number of epochs of training")
    parser.add_argument("--folder", type=str, default="/home/ivan/Experiments/JAN27-FACE-PIXELED-COLORS", help="")
    parser.add_argument("--data", type=str, default="/home/ivan/Datasets/JAN30-NPY-DATASET/train",
                        help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--height", type=int, default=256, help="size of image height")
    parser.add_argument("--workers", type=int, default=4, help="size of image height")
    parser.add_argument("--width", type=int, default=256, help="size of i mage width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500,
                        help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint", type=str, default="facades", help="name of the dataset")

    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    args = parser.parse_args()

    device = "cuda"
    size = (args.height, args.width)
    lr, betas = args.lr, (args.b1, args.b2)
    imgs_channels = args.channels
    batch_size = args.batch_size
    dataset_path = args.data
    num_epochs = args.epochs
    num_workers = args.workers
    save_folder = args.folder
    prefix = args.checkpoint
    os.makedirs(save_folder, exist_ok=True)

    load_size = (size[0]+24, size[1]+24)
    loader_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(3),
        transforms.Resize(load_size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[.5 for _ in range(9)], std=[.5 for _ in range(9)])
    ])
    dataset = get_dataset(dataset_path,
                          num_src_channels=imgs_channels,
                          num_total_channels=9,
                          transforms=loader_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    modules = {"G": GeneratorUNet(), "D": Discriminator()}
    if len(os.listdir(save_folder)):
        for net_type in modules:
            path = os.path.join(save_folder, f'{prefix}-{net_type}.ckpt')
            if os.path.isfile(path):
                modules[net_type].load_state_dict(torch.load(path))
                print("LOADED")
    else:
        initializer_fuct = WeightInitializer("kaiming")
        for net_type in modules:
            modules[net_type].apply(initializer_fuct)

    train(save_folder=save_folder,
          num_epochs=num_epochs,
          lr=lr,
          betas=betas,
          modules=modules,
          dataloader=dataloader,
          save_interval=1,
          device="cuda",
          lambda_rec=100)
