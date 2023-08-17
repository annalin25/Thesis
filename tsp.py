"""Defines the main task for the TSP

The TSP is defined by the following traits:
    1. Each city in the list must be visited once and only once
    2. The salesman must return to the original node at the end of the tour

Since the TSP doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in trainer.py to be None

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from helper import *



class TSPDataset(Dataset):    
    def __init__(self, route_list, mode, max=48): # mode = 'train', 'test'
        super(TSPDataset, self).__init__()

        # route_list = len(high) > 2718
        all_matrix = []
        all_order = []
        for i in tqdm(route_list, position = 0, leave = True):  # , position = 0, leave = True
            # mat, order = zone_mat_order_pad(i)
            mat, order = zone_mat_order_pad2(i, mode, max)
            mat2 = torch.unsqueeze(mat, 0)
            # order need unsqueeze?

            all_matrix.append(mat2)
            all_order.append(order)

        self.dataset = torch.cat(all_matrix, dim=0)
        self.dynamic = torch.zeros(len(route_list), 1, max)
        self.order = torch.tensor(all_order)
        self.route_id = route_list
        
        self.num_samples = len(route_list)
        self.num_zone = max

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [], self.order[idx], self.route_id[idx])
        # zone_actual_seq


def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask


def reward(static, tour_indices):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1).detach()

