from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import pytorch_lightning as pl
import torch, math, os
from torch.utils.data import Dataset, DataLoader
import numpy as np

def resolve_path_gdrive(relativePath):
    if os.path.exists('/content/drive'):
        return '/content/drive/MyDrive/work/gdrive-workspaces/git/nn_catalyst/' + relativePath
    else:
        from utils import get_project_root
        return get_project_root() + "/" + relativePath

class CatalystDataset(Dataset):

    def __init__(self, datafile='src/pl/merged_data_last29.csv'):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt(resolve_path_gdrive(datafile), delimiter=',', skiprows=1, max_rows=10, dtype=float)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:,:-29])  # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:,-29:-28])  # size [n_samples, 1]
        print(self.y_data)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class CatalystDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.entire_dataset = CatalystDataset()

    def prepare_data(self):
        train_set_size = int(len(self.entire_dataset) * 0.8)
        test_set_size = int(len(self.entire_dataset) * 0.1)
        valid_set_size = len(self.entire_dataset) - train_set_size - test_set_size
        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.entire_dataset, [train_set_size, valid_set_size, test_set_size])
        return

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
