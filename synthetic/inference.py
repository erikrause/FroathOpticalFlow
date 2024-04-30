from argparse import ArgumentParser

import torch

from synthetic.wrapper import OreSegmentor
import os
from synthetic.data import SyntheticRocks
from torch.utils.data import DataLoader
from synthetic import utils
import cv2
import numpy as np


def _get_testloader(config):
    testset = SyntheticRocks(color_roots=config["test_roots"]["colors"],
                            segments_roots=config["test_roots"]["segments"],
                            crop_resolution=config["test_random_crop_resolution"])
    testloader = DataLoader(dataset=testset,
                           batch_size=1,
                           shuffle=False)
    return testloader


def inference(args):
    config = utils.get_config(args.config_dir)
    wrapper = OreSegmentor.load_from_checkpoint(args.checkpoint_dir, learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    model = wrapper._model
    model = model.to("cuda")
    model = model.eval()
    testloader = _get_testloader(config)

    with torch.no_grad():
        test_iterator = iter(testloader)
        for i in range(args.n_frames):
            print(f"Frame {i}")
            colors, segments = next(test_iterator)
            pred = wrapper._sigmoid(model(colors.cuda()))
            pred_png = np.round((pred[0] * 255).cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
            segments_png = np.round((segments[0] * 255).cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
            cv2.imwrite(f"{args.out_dir}/{i}-pred.png", pred_png)
            cv2.imwrite(f"{args.out_dir}/{i}-GT.png", segments_png)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=10, help="how many frames to process")
    parser.add_argument("--out_dir", type=str, default=f"{os.getcwd()}/inference", help="output directory")
    parser.add_argument("--checkpoint_dir", type=str, default=os.getcwd(), help="where checkpoints are stored")
    parser.add_argument("--config_dir", type=str, default="synthetic/config.yaml", help="path to config.yml")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    inference(args)
