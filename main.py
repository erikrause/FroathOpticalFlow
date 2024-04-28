from argparse import ArgumentParser
from synthetic.preprocess import process_segments, split_set
import os
from synthetic.train import train


#names_preprocess("datasets/synthetic_rocks/frames/0")
# segments_sequential_mapping = get_segments_sequential_mapping("datasets/synthetic_rocks/segmentation/0")


preprocess_segments = False
split_test = False

if preprocess_segments:
    segmentation_pathes = ["datasets/synthetic_rocks/segmentation/0/sequence.0/",
                           "datasets/synthetic_rocks/segmentation/1/sequence.0/"]
    for segmentation_path in segmentation_pathes:
        process_segments(segmentation_path,
                         f"{segmentation_path}/preprocessed")

if split_test:
    color_pathes = ["datasets/synthetic_rocks/frames/0",
                    "datasets/synthetic_rocks/frames/1"]
    segmentation_pathes = ["datasets/synthetic_rocks/segmentation/0/sequence.0-preprocessed/",
                           "datasets/synthetic_rocks/segmentation/1/sequence.0-preprocessed/"]
    split_set(color_pathes, segmentation_pathes)

