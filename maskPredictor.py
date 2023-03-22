import os
import numpy as np
import torch

from models.create_model import evalModel

class MaskPredictor(object):
    """Class for generating masks from video"""
    def __init__(self, pretrainedWeightsPath, net='unet'):
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
            result = a.cpu().detach().numpy()
            for i in range(result.shape[0]):
                result[i][0] = (result[i][0] > th).astype(np.uint8)
        return result


    def get_mask_generator(self, videoGen, th = 0.7):
        """
            Produces masks frames from input generator.
            Parameters:
                :videoGen (generator) - generator from video
            Returns:
                :(np.ndarray, np.ndarray) - tuple of mask and source image
            """
        for images in videoGen:
            masks = self.compute_result(images[0], th)
            for  mask, frame, source in zip(masks, images[1], images[2]):
                yield mask[0], frame, source

    def get_mask_gt_generator(self, folderGen, th = 0.7):
        """
            Produces masks frames from input generator.
            Parameters:
                :videoGen (generator) - generator from video
            Returns:
                :(np.ndarray, np.ndarray) - tuple of mask and source image
            """
        for images in folderGen:
            masks = self.compute_result(images[0], th)
            for  mask, frame, source, gt in zip(masks, images[1], images[2], images[3]):
                yield mask[0], frame, source, gt

