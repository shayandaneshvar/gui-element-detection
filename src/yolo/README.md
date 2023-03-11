## YOLOv5
- created yolo5_full folder containing all images and yolo5_format files
- to make yolo work, created dataset folder with the following structure:

├── dataset.yaml

├── images

    ├── train
    
    ├── validation

    └── test

└── labels

    ├── train

    ├── validation

    └── test

- dataset.yaml content:

    train: ../dataset/images/train
    
    val: ../dataset/images/validation
    
    test: ../dataset/images/test
    
    nc: 12
    
    names: ['BackgroundImage', 'CheckedTextView', 'Icon', 'EditText', 'Image', 'Text', 'TextButton', 'Drawer', 'PageIndicator', 'UpperTaskBar', 'Modal', 'Switch']

- Ran initial YOLOv5 sections to populate the folders, and be ready for YOLOv5
- Copy dataset.yaml to yolov5 cloned folder