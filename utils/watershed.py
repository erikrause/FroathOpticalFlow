import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

class NoMarkersError(Exception):
    pass


import logging
import numpy as np


# Homomorphic filter class
class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.

    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H

        .
    """

    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = 1 / (1 + (Duv / filter_params[0] ** 2) ** filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = np.exp((-Duv / (2 * (filter_params[0]) ** 2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b * H) * I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H=None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter == 'butterworth':
            H = self.__butterworth_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'gaussian':
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt)) - 1
        return np.uint8(I)


# End of class HomomorphicFilter


def crop(frame):
    return frame[850:1750, 0:1000]


def preprocessing(frame):
    frame = frame.copy()
    frame = cv2.GaussianBlur(frame, (15, 15), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge([h, s, v])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray

def getSobel(v):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(v, ddepth, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    grad_y = cv2.Scharr(v,ddepth,0,1)
    grad_y = cv2.Sobel(v, ddepth, 0, 1, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    maskSobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    maskSobel = cv2.threshold(maskSobel, 150, 255, cv2.THRESH_BINARY)[1]
#     print('sobel')
#     show(maskSobel)
    return maskSobel

def findMarkers(v):
    kernel = np.ones((9, 9), np.uint8)
    v = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    v = cv2.pyrMeanShiftFiltering(v, 11, 21)
    v = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
    homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)
    v = homo_filter.filter(I=v, filter_params=[20,2])
#     print('before markers')
#     show(v)
#     thresh = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    thresh = cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel = np.ones((15, 15), np.uint8)
    markers = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    markers = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     print('markers')
#     show(markers)
    return thresh
#     return markers

def doWatershed(markers, maskSobel):
    distance_map = ndimage.distance_transform_edt(markers)
    local_max = peak_local_max(distance_map, indices=False, min_distance=10, labels=markers)
#     print('local max')
#     show(local_max)
    # Perform connected component analysis then apply Watershed
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]

    labels = watershed(-distance_map, markers)#, mask=maskSobel)
    return labels



def findContours(gray, mode=3, returnContours=False):
    markers = findMarkers(gray)
    if np.all(np.count_nonzero(markers == 255) < 100000):
        raise NoMarkersError
    maskSobel = getSobel(gray)
    # Iterate through unique labels
    labels = doWatershed(markers, maskSobel)

    black = np.zeros(gray.shape, np.uint8)

    n = 0
    Areas = []
    contours = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # Create a mask
        mask = np.zeros(markers.shape, dtype="uint8")
        mask[labels == label] = 255
        # Find contours and determine contour area
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key=cv2.contourArea)
        contours.append(c)
        area = cv2.contourArea(c)
        if area > 10:
            Areas.append(area)
        n += 1
        if mode == 0:
            cv2.drawContours(black, [c], -1, (255, 255, 255), -1)
        if mode == 1:
            cv2.drawContours(black, [c], -1, (0, 0, 0), 1)
        if mode == 2:
            cv2.drawContours(black, [c], -1, (255, 255, 255), -1)
            cv2.drawContours(black, [c], -1, (0, 0, 0), 2)
        if mode == 3:
            cv2.drawContours(gray, [c], -1, (0, 0, 0), 1)
    if returnContours:
        return contours
    else:
        return gray, black