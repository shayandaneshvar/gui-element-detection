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
- Copy dataset.yaml to yolov5 cloned folder, and update the directories accordingly
- Run the model for 2 and 100 epochs
- Run the trained model on training data (available in the runs/detect folder of yolov5)
- Achieved slightly better (less than 1% overall) than the paper as latest version of Yolov5 (7th edition) is used

## YOLOv8
- Same author as YoloV5
- ran the same dataset using their cli 
- achieves slightly worse performance even when trained for 3 times the epoch i.e. 300 epochs
- ran test to predict the tests
- Moving to Yolov7 from another author and hoping that it works better :|
- Why YOLOv8 does worse than Yolov5 in GUI element collection? 
  - Based on the code and what author states in a github issue, YOLOv8 is anchor free
  - It has to do with the fact that anchor-based methods do better as they try to predict a bounding box instead of the objects' center
  - This makes sense as most GUI elements are rectangular to some extent, e.g. images, buttons, texts, etc
  - But not sure if this is exactly the reason
  - YOLOv8 paper will be out after they take care of some fancy deployment capability with some other platforms, so not soon!

### YOLOv7
