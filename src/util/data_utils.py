import os
import random
from enum import Enum
from glob import glob

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


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


def merge_bounding_boxes(arrays,classes=13):
    items = arrays.shape[0] if len(arrays.shape) == 4 else 1
    merged = []
    for i in range(items):
        if items == 1:
            current = arrays
        else:
            current = arrays[i,]

        current = current * np.array([[np.arange(classes)]])
        merged.append(np.max(current, axis=2, out=None))

    merged = np.array(merged)
    if items == 1:
        return merged[0]
    else:
        return merged


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
            # cv2.imshow('sds', image)
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
        # print(label_path)
        labels_mask.append(mask)
        # cv2.imshow("sd", mask/1.0)
        # cv2.waitKey()

    return np.array(labels_mask)  # of shape (items,(shape), classes + 1 bg)


class LazyDataLoader(Sequence):
    def __init__(self, split: Splits, images_path, labels_path, batch_size, n_classes=13, resize=True,
                 shape=(416, 416), normalize=True):
        self.image_path = glob(os.path.join(images_path, split.get_path()) + "*.jpg")
        self.mask_path = glob(os.path.join(labels_path, split.get_path()) + "*.txt")
        self.image_path.sort()
        self.mask_path.sort()
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.index = 0
        self.resize = resize
        self.shape = shape
        self.normalize = normalize

    # def __check_index__(self):
    #     return self.index

    # def __update_index__(self):
    #     self.index = self.index + self.batch_size

    # Steps per epoch
    def __len__(self):

        # maybe something to confirm equal lengths
        assert len(self.image_path) == len(self.mask_path)

        return len(self.image_path) // self.batch_size

        # Shuffles the images and masks **together**

    def on_epoch_end(self):

        combined = list(zip(self.image_path, self.mask_path))

        random.shuffle(combined)

        self.image_path[:], self.mask_path[:] = zip(*combined)

    # Generates data, feeds to training
    def __getitem__(self, index):

        # index = self.__check_index__()

        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        if end > len(self.image_path):
            end = len(self.image_path)
            print("Updating Last Batch end")

        # loading images
        images = []
        for image_path in self.image_path[start:end]:
            image = cv2.imread(image_path)
            if self.resize:
                image = cv2.resize(image, self.shape)
                # cv2.imshow('sds', image)
            if self.normalize:
                image = image / 255.0
            images.append(image)

        labels_masks = []
        # loading labels and transforming into batches
        for label_path in self.mask_path[start:end]:
            mask = np.zeros(self.shape, dtype="int")
            with open(label_path) as file:
                lines = [line.rstrip() for line in file]
                for line in lines:
                    line_splits = line.split(" ")
                    cls = int(line_splits[0])
                    center_x = float(line_splits[1])
                    center_y = float(line_splits[2])
                    width = float(line_splits[3])
                    height = float(line_splits[4])
                    start_x = max(int(self.shape[1] * (center_x - (width / 2))), 0)
                    start_y = max(int(self.shape[0] * (center_y - height / 2)), 0)
                    end_x = min(int(self.shape[1] * (center_x + (width / 2))), self.shape[1] - 1)
                    end_y = min(int(self.shape[0] * (center_y + height / 2)), self.shape[1] - 1)

                    mask[start_y:end_y:, start_x] = cls + 1
                    mask[start_y:end_y:, end_x] = cls + 1
                    mask[start_y, start_x:end_x:] = cls + 1
                    mask[end_y, start_x:end_x:] = cls + 1
                    # 0 is a class, but as mask 0 must be background which is not a class (in the dataset)
            labels_masks.append(mask)

        # self.__update__index()
        batch_x = np.array(images).astype('float32')
        batch_y = to_categorical(np.array(labels_masks).astype('float32'), num_classes=self.n_classes)
        print(batch_x.shape)
        print(batch_y.shape)
        return (batch_x, batch_y)


class LazyDataLoaderV2(LazyDataLoader):
    """
    same as V1 but V1 only segments the bounding box with width of 1px. V2 however, segments insides of the bb as well
    """
    def __init__(self, split: Splits, images_path, labels_path, batch_size, n_classes=13, resize=True,
                 shape=(416, 416), normalize=True):
        super().__init__(split, images_path, labels_path, batch_size, n_classes, resize, shape, normalize)

    def __getitem__(self, index):

        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        if end > len(self.image_path):
            end = len(self.image_path)
            print("Updating Last Batch end")

        # loading images
        images = []
        for image_path in self.image_path[start:end]:
            image = cv2.imread(image_path)
            if self.resize:
                image = cv2.resize(image, self.shape)
                # cv2.imshow('sds', image)
            if self.normalize:
                image = image / 255.0
            images.append(image)

        labels_masks = []
        # loading labels and transforming into batches
        for label_path in self.mask_path[start:end]:
            mask = np.zeros(self.shape, dtype="int")
            with open(label_path) as file:
                lines = [line.rstrip() for line in file]
                for line in lines:
                    line_splits = line.split(" ")
                    cls = int(line_splits[0])
                    center_x = float(line_splits[1])
                    center_y = float(line_splits[2])
                    width = float(line_splits[3])
                    height = float(line_splits[4])
                    start_x = max(int(self.shape[1] * (center_x - (width / 2))), 0)
                    start_y = max(int(self.shape[0] * (center_y - height / 2)), 0)
                    end_x = min(int(self.shape[1] * (center_x + (width / 2))), self.shape[1] - 1)
                    end_y = min(int(self.shape[0] * (center_y + height / 2)), self.shape[1] - 1)

                    mask[start_y:end_y:, start_x:end_x:] = cls + 1
                    # 0 is a class, but as mask 0 must be background which is not a class (in the dataset)
            labels_masks.append(mask)

        batch_x = np.array(images).astype('float32')
        batch_y = to_categorical(np.array(labels_masks).astype('float32'), num_classes=self.n_classes)
        return (batch_x, batch_y)


if __name__ == '__main__':
    print("This should be used for testing only!")
    import sys

    sys.path.insert(1,
                    'E:/Research/course research projects/data-driven software engineering/gui-element-detection/src')
    from constants import VINS_MERGED_YOLO_SPLITS_LABELS as ds_y_path

    # print(ds_path)
    # print(len(load_data(Splits.TRAIN, ds_path,resize=True, shape=())))
    # print(os.path.join(ds_path, 'train/'))
    load_labels_as_masks(split=Splits.TRAIN, path=ds_y_path)
