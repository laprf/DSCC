"""Utility helpers for coordinate generation and point indexing."""

import torch
import torch.nn.functional as F


def cal_coords(grid_size, img_size=512):
    """Return sorted grid-center coordinates and dense pixel coordinates."""

    grid_step = img_size / grid_size
    grid_indices = torch.arange(grid_size)
    center_coords = (grid_indices + 0.5) * grid_step

    # center coords
    center_x, center_y = torch.meshgrid(center_coords, center_coords, indexing='ij')
    center_x = torch.round(center_x.reshape(-1)).long()
    center_y = torch.round(center_y.reshape(-1)).long()
    sorted_coords = torch.tensor(sorted(zip(center_x.tolist(), center_y.tolist()), key=lambda coord: (coord[0], coord[1])))

    # pixel coords
    pixel_x, pixel_y = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing='ij')
    pixel_coords = torch.stack([pixel_x.reshape(-1), pixel_y.reshape(-1)], dim=1)

    return sorted_coords.float(), pixel_coords.float()


def index_points(x, idx):
    """Gather point features from x using batched indices."""

    x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
    return x  # [B, M - r, D]

def _normalize(x):
    B = x.shape[0]
    min_val = x.reshape(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
    max_val = x.reshape(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
    return (x - min_val) / (max_val - min_val + 1e-8)
