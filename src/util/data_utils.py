import os
from enum import Enum
from glob import glob

import cv2
import numpy as np

from tensorflow.keras.utils import to_categorical
class Splits(Enum):
    TRAIN = 1
    TEST = 2
    VALIDATION = 3

    def get_path(self):
        if self == Splits.TRAIN:
            return 'train/'
        elif self == Splits.TEST:
            return 'test/'
        elif self == Splits.VALIDATION:
            return 'validation/'


def load_data(split: Splits, path=None, resize=False, shape=(416, 416), normalize=True):
    """
    :param split:
    :param path:
    :param resize:
    :param shape: (height, width)
    :param normalize:
    :return:
    """
    path = os.path.join(path, split.get_path())
    image_paths = glob(path + '*.jpg')
    image_paths.sort()
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if resize:
            image = cv2.resize(image, shape)
            cv2.imshow('sds', image)
        if normalize:
            image = image / 255.0
            images.append(image)
    print(f"Split {split.name} data was loaded with {len(images)} images")
    return np.array(images)


def load_labels_as_masks(split: Splits, path=None, shape=(416, 416)):
    """

    :param split:
    :param path:
    :param shape: (height, width)
    :return: masks
    """
    path = os.path.join(path, split.get_path())
    label_paths = glob(path + '*.txt')
    label_paths.sort()
    labels_mask = []
    for label_path in label_paths:
        mask = np.zeros(shape, dtype="int")
        with open(label_path) as file:
            lines = [line.rstrip() for line in file]
            for line in lines:
                line_splits = line.split(" ")
                cls = int(line_splits[0])
                center_x = float(line_splits[1])
                center_y = float(line_splits[2])
                width = float(line_splits[3])
                height = float(line_splits[4])
                start_x = max(int(shape[1] * (center_x - (width / 2))), 0)
                start_y = max(int(shape[0] * (center_y - height / 2)), 0)
                end_x = min(int(shape[1] * (center_x + (width / 2))), shape[1] - 1)
                end_y = min(int(shape[0] * (center_y + height / 2)), shape[1] - 1)

                mask[start_y:end_y:, start_x] = cls + 1
                mask[start_y:end_y:, end_x] = cls + 1
                mask[start_y, start_x:end_x:] = cls + 1
                mask[end_y, start_x:end_x:] = cls + 1
                # 0 is a class, but as mask 0 must be background which is not a class
        print(label_path)
        labels_mask.append(mask)
        # cv2.imshow("sd", mask/1.0)
        # cv2.waitKey()

    return np.array(labels_mask) # of shape (items,(shape), classes + 1 bg)


if __name__ == '__main__':
    print("This should be used for testing only!")
    import sys
    sys.path.insert(1, 'E:/Research/course research projects/data-driven software engineering/gui-element-detection/src')
    from constants import VINS_MERGED_YOLO_SPLITS_LABELS as ds_y_path

    # print(ds_path)
    # print(len(load_data(Splits.TRAIN, ds_path,resize=True, shape=())))
    # print(os.path.join(ds_path, 'train/'))
    load_labels_as_masks(split=Splits.TRAIN, path=ds_y_path)
