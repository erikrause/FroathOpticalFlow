import os
import cv2
import numpy as np


class DataManager(object):
    """Class for generating batches of frames from video"""

    def __init__(self, videoPath):
        """
        Constructor
        Parameters:
            :videoPath(str) - path for the input video
        Returns:
        """
        if os.path.exists(videoPath):
            self.video = cv2.VideoCapture(videoPath)
            self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        else:
            raise FileNotFoundError('No video file: '+videoPath)

    def preprocessing(self, frame):
        """
         Produce input for neural network.
         Parameters:
            :frame (np.ndarray): source image,
         Returns:
            :np.ndarray - contrast gray image
         """
        import cv2
        source = frame.copy()
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        img = cv2.resize(gray, (256, 256))
        # img = img/255.
        img = img.reshape(1, 256, 256)
        return img, gray, source

## TODO: fix for normal pytorch batches
    def get_frames_gen(self, batch_size=32, num=np.inf):
        """
          Batches generator for neural network input.
          Parameters:
              :batch_size(int) - size of the images batch for Unet.
              :num(int) - maximum number of frames.
          """
        counter = 0
        while self.video.isOpened() and counter < num:
            imgs = []
            imgs_gray = []
            sources = []
            for i in range(batch_size):
                ret, img = self.video.read()
                if not ret:
                    break
                img, img_gray, source = self.preprocessing(img)
                imgs.append(img)
                imgs_gray.append(img_gray)
                sources.append(source)
                counter += 1
            yield np.asarray(imgs), np.asarray(imgs_gray), np.asarray(sources)
        self.video.release()

