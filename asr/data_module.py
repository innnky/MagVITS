import pytorch_lightning as pl

from asr.meldataset import build_dataloader



def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    return train_list, val_list

class ASRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='dump/', batch_size=64,num_workers=8):
        super().__init__()
        train_path = f'{data_dir}/train_files.list'
        val_path = f'{data_dir}/val_files.list'
        train_list, val_list = get_data_path_list(train_path, val_path)
        train_dataloader = build_dataloader(train_list,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            dataset_config={})

        val_dataloader = build_dataloader(val_list,
                                          batch_size=batch_size,
                                          validation=True,
                                          num_workers=1,
                                          dataset_config={})
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader


    def train_dataloader(self):

        return self.train_loader

    def val_dataloader(self):

        return self.val_loader