import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import re
import albumentations as A
import typing
from models.unet import preprocessing_fn


def names_preprocess(color_root):
    color_pathes = sorted(glob.glob(f"{color_root}/*.png"), key=get_frame_frame_id)
    os.mkdir(f"{color_root}-processed")

    for frame_id in range(len(color_pathes)):
        #re.match(r"\d+(?=-0)", color_pathes[frame_id])
        color = cv2.imread(color_pathes[frame_id])

        cv2.imwrite(f"{color_root}-processed/{frame_id}.png", color)


def get_frame_frame_id(path):
    return int(re.findall(r"\d+(?=-0)", path)[0])


def get_segment_frame_id(path):
    return int(re.findall(r"\d+(?=\.camera)", path)[0])


# Returns mapping from sequential id to segment's step
def get_segments_sequential_mapping(segments_root):
    segments_pathes = sorted(glob.glob(f"{segments_root}/sequence.0/*.png"), key=get_segment_frame_id)

    frames_map = {}
    for frame_id in range(len(segments_pathes)):
        n_frame_id = int(re.findall(r"\d+(?=\.)", segments_pathes[frame_id])[0])
        if n_frame_id != frame_id + 1:
            print(frame_id + 1)
            frames_map[frame_id] = n_frame_id


def get_transform(crop_resolution: tuple[int, int]):

    return A.Compose([
        A.RandomCrop(crop_resolution[0], crop_resolution[1], p=1.0),
        A.RandomRotate90(p=1.0),
        A.Flip()])
        # TODO: random rotate all degrees
        # TODO: random global brightness
        # TODO: random rescale+crop


class SyntheticRocks(Dataset):
    def __init__(self,
                 color_roots: list[str],
                 segments_roots: list[str],
                 crop_resolution: typing.Optional[tuple[int, int]] = None,
                 #device=torch.device("cpu")
                 ):
        self.crop_resolution = crop_resolution

        assert len(color_roots) == len(segments_roots)
        self.color_pathes = []
        self.segments_pathes = []
        for i in range(len(color_roots)):
            color_root = color_roots[i]
            segments_root = segments_roots[i]
            assert os.path.isdir(color_root) and os.path.isdir(segments_root)
            self.color_pathes.extend(sorted(glob.glob(f"{color_root}/*.png"), key=get_frame_frame_id))
            self.segments_pathes.extend(sorted(glob.glob(f"{segments_root}/*.png"), key=get_segment_frame_id))

        assert len(self.color_pathes) == len(self.segments_pathes)
        for i in range(len(self.color_pathes)):
            assert os.path.exists(self.color_pathes[i])
            assert os.path.exists(self.segments_pathes[i])

        if crop_resolution:
            self._transform = get_transform(crop_resolution=crop_resolution)
        else:
            self._transform = None
        self.preprocessing_fn = preprocessing_fn

    def __len__(self):
        return len(self.color_pathes)

    # def __getitem__(self, idx):
    #     color = cv2.imread(self.color_pathes[idx]) / 255.
    #     segments = cv2.imread(self.segments_pathes[idx]) / 255
    #
    #     if self._transform:
    #         transformed = self._transform(color=color, segments=segments)
    #         color = transformed["color"]
    #         segments = transformed["segments"]
    #
    #     # changing channel axis order from cv2-like to pytorch-like
    #     color = np.moveaxis(color, -1, 0)
    #     segments = np.expand_dims(segments, 0)
    #
    #     return color, segments

    def __getitem__(self, idx):
        color = cv2.imread(self.color_pathes[idx])[..., ::-1]
        segments = cv2.imread(self.segments_pathes[idx], cv2.IMREAD_UNCHANGED) / 255.
        color = self.preprocessing_fn(color)

        if self._transform:
            transformed = self._transform(image=color, mask=segments)
            color = transformed["image"]
            segments = transformed["mask"]

        # changing channel axis order from cv2-like to pytorch-like
        color = np.moveaxis(color, -1, 0)
        segments = np.expand_dims(segments, 0)

        return color.astype(np.float32), segments.astype(np.float32)

