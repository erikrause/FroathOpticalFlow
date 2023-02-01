import os

import numpy as np
import torch

from models.create_model import evalModel

class MaskPredictor(object):
    """Class for generating masks from video"""
    def __init__(self, pretrainedWeightsPath):
        """
        Constructor of mask generator.
        Parameters:
            :pretrainedWeightsPath (str) - path for pretrained weights.
        """
        if os.path.exists(pretrainedWeightsPath):
            self.model = evalModel(pretrainedWeightsPath)
        else:
            raise FileNotFoundError('No weights file: ' + pretrainedWeightsPath)

    def compute_result(self, source, th=0.5):
        with torch.no_grad():
            a = self.model(torch.from_numpy(source).type(torch.FloatTensor) / 255)
            pred = (a.cpu().detach().numpy()[0][0] > th).astype(np.uint8)
        return pred

    def get_mask_generator(self, videoGen):
        """
            Produces masks frames from input generator.
            Parameters:
                :videoGen (generator) - generator from video
            Returns:
                :(np.ndarray, np.ndarray) - tuple of mask and source image
            """
        for images in videoGen:
             for mask, frame, source in zip(self.compute_result(images[0]), images[1], images[2]):
                yield mask, frame, source


