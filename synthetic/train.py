from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from synthetic.data import SyntheticRocks
from torch.utils.data import DataLoader
from synthetic.wrapper import OreSegmentor
from synthetic import utils
import os
from synthetic.data import names_preprocess, get_segments_sequential_mapping


def train(args):
    config = utils.get_config(args.config_dir)

    os.makedirs(config["tensorboard_root"], exist_ok=True)
    logger = TensorBoardLogger(save_dir=config["tensorboard_root"])
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, every_n_epochs=1)

    trainset = SyntheticRocks(color_roots=config["train_roots"]["colors"],
                              segments_roots=config["train_roots"]["segments"],
                              crop_resolution=config["train_random_crop_resolution"])
    trainloader = DataLoader(dataset=trainset,
                             batch_size=config["batch_size"],
                             shuffle=True)
    valset = SyntheticRocks(color_roots=config["val_roots"]["colors"],
                            segments_roots=config["val_roots"]["segments"],
                            crop_resolution=config["val_random_crop_resolution"])
    valloader = DataLoader(dataset=valset,
                           batch_size=1,
                           shuffle=False)

    wrapper = OreSegmentor(learning_rate=config["learning_rate"],
                           weight_decay=config["weight_decay"])
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config["n_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(wrapper, trainloader, valloader, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint file to restore model from")
    parser.add_argument("--checkpoint_dir", type=str, default=os.getcwd(), help="where to store checkpoints")
    parser.add_argument("--config_dir", type=str, default="synthetic/config.yaml", help="path to config.yml")
    args = parser.parse_args()
    train(args)
