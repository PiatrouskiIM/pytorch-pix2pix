from ops.pix2pix_objective import Pix2pixObjective
import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader


@torch.no_grad()
def test_pix2pix(dataset, coach, modules, device="cpu", **kwargs):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    epoch_stats = []
    for i_batch, data in enumerate(dataloader):
        data = (data[:, 0], data[:, 1:].reshape(data.size(0), -1, data.size(-2), data.size(-1)))
        batch_stats = {}
        _, stats = coach.practice_discrimination(modules["G"], modules["D"], data, device)
        batch_stats["D"] = stats

        loss, stats = coach.practice_generation(modules["G"], modules["D"], data, device)
        batch_stats["G"] = stats

        epoch_stats.append(batch_stats)
    return epoch_stats

if __name__ == "__main__":
    raise NotImplementedError
