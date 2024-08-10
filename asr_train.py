from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from asr.data_module import ASRDataModule
from asr.meldataset import build_dataloader
from utils import *
from asr.models import build_model
from asr.trainer import ASRTrainer

import os
import os.path as osp
import yaml
import shutil
import click

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
def build_criterion(critic_params={}):
    criterion = {
        "ce": nn.CrossEntropyLoss(ignore_index=-1),
        "ctc": torch.nn.CTCLoss(**critic_params.get('ctc', {})),
    }
    return criterion

@click.command()
@click.option('-c', '--config_path', default='configs/asr.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    log_dir = config['log_dir']
    os.makedirs(log_dir,exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))


    batch_size = config.get('batch_size', 10)
    epochs = config.get('epochs', 1000)
    save_freq = config.get('save_freq', 20)
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)

    # train_list, val_list = get_data_path_list(train_path, val_path)
    # train_dataloader = build_dataloader(train_list,
    #                                     batch_size=batch_size,
    #                                     num_workers=8,
    #                                     dataset_config=config.get('dataset_params', {}))
    #
    # val_dataloader = build_dataloader(val_list,
    #                                   batch_size=batch_size,
    #                                   validation=True,
    #                                   num_workers=2,
    #                                   dataset_config=config.get('dataset_params', {}))
    data_module = ASRDataModule(data_dir='dump', batch_size=batch_size, num_workers=8)

    model = build_model(model_params=config['model_params'] or {})

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename=('{epoch}-{step}'),
        every_n_train_steps=None,
        every_n_epochs=1,
        verbose=True,
        save_last=True
    )
    logger = WandbLogger(project="asr-align")

    blank_index = 0

    criterion = build_criterion(critic_params={
        'ctc': {'blank': blank_index},
    })
    training_wrapper = ASRTrainer(model=model, criterion=criterion,mono_start_epoch=10,lr=1e-4)
    if config.get('pretrained_model',None):
        training_wrapper.load_checkpoint(config['pretrained_model'])

    trainer: Trainer = Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=-1,
        benchmark=False,
        fast_dev_run=False,
        strategy=DDPStrategy(),
        logger=logger,
        callbacks=[checkpoint_callback])

    trainer.fit(training_wrapper, data_module)


if __name__ == "__main__":
    main()