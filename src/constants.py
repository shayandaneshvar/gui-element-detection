import glob

# project constants

VINS_DATASET_PATH = 'E:/Research/course research projects/data-driven software engineering/datasets/All Dataset/'

VINS_ANDROID = f"{VINS_DATASET_PATH}Android"
VINS_RICO = f"{VINS_DATASET_PATH}Rico"
VINS_IPHONE = f"{VINS_DATASET_PATH}Iphone"
VINS_UPLABS = f"{VINS_DATASET_PATH}uplabs"
VINS_WIREFRAMES = f"{VINS_DATASET_PATH}Wireframes"

## what was done in the yolov5 paper and the paper before that?

VINS_MERGED = f"{VINS_DATASET_PATH}Merged"
VINS_MERGED_IMAGES = f"{VINS_MERGED}/images/"
VINS_MERGED_ANNOTATIONS = f"{VINS_MERGED}/annotations/"

if __name__ == '__main__':
    images = glob.glob(VINS_MERGED_IMAGES + '*.jpg')
    annotations = glob.glob(VINS_MERGED_ANNOTATIONS + '*.xml')
    print(len(images))
    print(len(annotations))

    pxmls = [k.split('\\')[-1].split('.')[0] for k in annotations]
    imgs = [k.split('\\')[-1].split('.')[0] for k in images]
    # print(xmls)
    unclean = False

    for img in imgs:
        if img not in pxmls:
            print(f"{img} jpg file not found")

            unclean = True

    for pxml in pxmls:
        if pxml not in imgs:
            print(f"{pxml} xml file not found")
            unclean = True

    if unclean:
        raise AssertionError("UNCLEAN!")