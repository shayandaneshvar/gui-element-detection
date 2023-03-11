import glob

# project constants

VINS_DATASET_PATH = 'E:/Research/course research projects/data-driven software engineering/datasets/All Dataset/'

VINS_ANDROID = f"{VINS_DATASET_PATH}Android"
VINS_RICO = f"{VINS_DATASET_PATH}Rico"
VINS_IPHONE = f"{VINS_DATASET_PATH}Iphone"
VINS_UPLABS = f"{VINS_DATASET_PATH}uplabs"
VINS_WIREFRAMES = f"{VINS_DATASET_PATH}Wireframes"

# what was done in the yolov5 paper and the paper before that? all but the wireframes

VINS_MERGED = f"{VINS_DATASET_PATH}Merged"
VINS_MERGED_IMAGES = f"{VINS_MERGED}/images/"
VINS_MERGED_ANNOTATIONS = f"{VINS_MERGED}/annotations/"
VINS_MERGED_YOLO = f"{VINS_MERGED}/yolo5_format/"
# EditText -> InputField
# CheckedTextView -> CheckedView
# Drawer -> Sliding Menu
# Modal -> Pop-Up window
# Discarded Rare Items: Spinner, Card, Multi tab, Toolbar, Bottom_Navigation, Remember

# Assholes who came up with VINS, updated the names !!! WTF! Wasted 3 hours finding wtf is going on!
VINS_CLASSES = {'BackgroundImage': 0, 'CheckedTextView': 1, 'Icon': 2, 'EditText': 3,
                'Image': 4, 'Text': 5, 'TextButton': 6, 'Drawer': 7, 'PageIndicator': 8,
                'UpperTaskBar': 9, 'Modal': 10, 'Switch': 11}

VINS_CLASSES_COUNT = {'BackgroundImage': 0, 'CheckedTextView': 0, 'Icon': 0, 'EditText': 0,
                      'Image': 0, 'Text': 0, 'TextButton': 0, 'Drawer': 0, 'PageIndicator': 0,
                      'UpperTaskBar': 0, 'Modal': 0, 'Switch': 0}


def reset_classes_counts():
    global VINS_CLASSES_COUNT
    for key in VINS_CLASSES_COUNT.keys():
        VINS_CLASSES_COUNT[key] = 0


# Final Count:

# YOLOv5 Format per line:
# (class x_center y_center width height) classes start from 0, the rest are normalized

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
