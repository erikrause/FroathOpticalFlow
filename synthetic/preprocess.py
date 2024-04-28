import os
import glob
import cv2
import numpy as np
from synthetic.data import get_frame_frame_id, get_segment_frame_id


def old_test():
    img = cv2.imread("img-origin.png").astype(np.float32)

    ddepth = cv2.CV_32FC3
    # edges = cv2.Laplacian(img, ddepth, ksize=3)
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    edges = cv2.filter2D(img, ddepth, kernel)
    edges_gray = np.zeros((edges.shape[0], edges.shape[1]))
    edges_gray[:] = edges[..., 0] + edges[..., 1] + edges[..., 2]
    edges_gray[np.where(edges_gray != 0)] = 255
    edges_gray = edges_gray.astype(np.uint8)
    edges_gray = (edges_gray * -1) + 255  # reverse
    # abs_edges = cv2.convertScaleAbs(edges)
    # cv2.imshow("Result", abs_edges)
    # cv2.waitKey(0)
    #cv2.imwrite("img-origin.png", img)
    cv2.imwrite("img-edges.png", edges)
    cv2.imwrite("img-result.png", edges_gray)

# crop = img[146:149, 80:83, 1].astype(np.float32)
# edges_crop = cv2.filter2D(crop, ddepth, kernel)
# edge_crop = -4 * crop[1, 1] + crop[0, 1] + crop[1, 0] + crop[2, 1] + crop[1, 2]
# cv2.imwrite("img-crop.png", crop)
# cv2.imwrite("img-crop_edges.png", edges_crop)


def process_segments(root,
                     out_root,
                     ground_color=(255, 138, 0),
                     border_color=(0, 0, 0)):

    os.makedirs(out_root, exist_ok=True)
    segments_pathes = sorted(glob.glob(f"{root}/*.png"), key=get_segment_frame_id)

    for segments_path in segments_pathes:
        img_int = cv2.imread(segments_path)
        img = img_int.astype(np.float32)

        # Laplacian filtering
        ddepth = cv2.CV_32FC3
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]])
        edges = cv2.filter2D(img, ddepth, kernel)
        edges_gray = np.zeros((edges.shape[0], edges.shape[1]))
        edges_gray[:] = edges[..., 0] + edges[..., 1] + edges[..., 2]
        edges_gray[np.where(edges_gray != 0)] = 255
        edges_gray = edges_gray.astype(np.uint8)
        edges_gray = (edges_gray * -1) + 255  # reverse

        # Make black all background and conveyor texture
        ground_color = np.array(ground_color, dtype=np.uint8)[::-1]     # RGB -> BGR
        border_color = np.array(border_color, dtype=np.uint8)[::-1]     # RGB -> BGR
        ground_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        border_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        ground_mask[np.where(np.all(img_int == ground_color, axis=-1))] = 255
        border_mask[np.where(np.all(img_int == border_color, axis=-1))] = 255
        edges_gray[np.where(ground_mask == 255)] = 0
        edges_gray[np.where(border_mask == 255)] = 0


        filename = os.path.basename(segments_path)
        cv2.imwrite(f"{out_root}/{filename}", edges_gray)


def split_set(color_roots,
              segments_roots):

    for i in range(len(color_roots)):
        color_root = color_roots[i]
        segments_root = segments_roots[i]
        color_pathes = sorted(glob.glob(f"{color_root}/*.png"), key=get_frame_frame_id)
        segments_pathes = sorted(glob.glob(f"{segments_root}/*.png"), key=get_segment_frame_id)

        os.makedirs(f"{color_root}/train/", exist_ok=True)
        os.makedirs(f"{segments_root}/train/", exist_ok=True)
        os.makedirs(f"{color_root}/valid/", exist_ok=True)
        os.makedirs(f"{segments_root}/valid/", exist_ok=True)
        os.makedirs(f"{color_root}/test/", exist_ok=True)
        os.makedirs(f"{segments_root}/test/", exist_ok=True)

        color_pathes_train = []
        segments_pathes_train = []
        color_pathes_valid = []
        segments_pathes_valid = []
        color_pathes_test = []
        segments_pathes_test = []

        # Valid (10%)
        color_pathes_valid = color_pathes[:332]
        color_pathes = color_pathes[332:]
        segments_pathes_valid = segments_pathes[:332]
        segments_pathes = segments_pathes[332:]

        # Test (10%)
        color_pathes_test = color_pathes[:332]
        color_pathes = color_pathes[332:]
        segments_pathes_test = segments_pathes[:332]
        segments_pathes = segments_pathes[332:]

        # Train (80%)
        color_pathes_train = color_pathes
        segments_pathes_train = segments_pathes

        for i in range(len(color_pathes_train)):
            color_basename = os.path.basename(color_pathes_train[i])
            segments_basename = os.path.basename(segments_pathes_train[i])
            os.rename(color_pathes_train[i], f"{color_root}/train/{color_basename}")
            os.rename(segments_pathes_train[i], f"{segments_root}/train/{segments_basename}")

        for i in range(len(color_pathes_valid)):
            color_basename = os.path.basename(color_pathes_valid[i])
            segments_basename = os.path.basename(segments_pathes_valid[i])
            os.rename(color_pathes_valid[i], f"{color_root}/valid/{color_basename}")
            os.rename(segments_pathes_valid[i], f"{segments_root}/valid/{segments_basename}")

        for i in range(len(color_pathes_test)):
            color_basename = os.path.basename(color_pathes_test[i])
            segments_basename = os.path.basename(segments_pathes_test[i])
            os.rename(color_pathes_test[i], f"{color_root}/test/{color_basename}")
            os.rename(segments_pathes_test[i], f"{segments_root}/test/{segments_basename}")
