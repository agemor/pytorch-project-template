import torch
import torch.utils as utils


class DummyDataset(utils.data.Dataset):

    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 0
