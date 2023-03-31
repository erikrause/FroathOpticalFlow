from watershed import findMarkers, getSobel, doWatershed, NoMarkersError
import numpy as np
import cv2


def drawMask(real, contours):
    black=np.zeros(real.shape, np.uint8)
    black = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
    cv2.drawContours(black,contours,-1,(255, 255, 255), -1)
    cv2.drawContours(black,contours,-1,(0, 0, 0), 3)
    return black

def drawContours(real, contours):
    cv2.drawContours(real, contours, -1, (0,200,0), 2)


def get_mask(img1, img2):

    from skimage.metrics import structural_similarity as compare_ssim
    import imutils

    grayA = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    #     (score, diff) = compare_ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 220, 255, cv2.THRESH_BINARY_INV)[1]
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    cv2.drawContours(thresh, contours, -1, (255, 255, 255), 5)
    cv2.drawContours(thresh, contours, -1, (255, 255, 255), -1)
    return thresh
