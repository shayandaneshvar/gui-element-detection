# Gui-element Detection using YOLOv5, YOLOv6, YOLOv7, and YOLOv8

[Link to the paper](https://arxiv.org/abs/2408.03507)

[VINS Dataset](https://github.com/sbunian/VINS)

## Related papers' repositories

- [74 - Object Detection for Graphical User Interface: Old Fashioned or Deep Learning or a Combination?](https://github.com/chenjshnn/Object-Detection-for-Graphical-User-Interface)
  (CenterNet, Faster-RCNN, YOLOv3, Xianyu)
- [72 - UIED: a hybrid tool for GUI element detection](https://github.com/MulongXie/UIED)
- [CenterNet v2](https://github.com/xingyizhou/CenterNet2)

### Important Papers

- 72 - UIED: a hybrid tool for GUI element detection
- 74 - Object detection for graphical user interface: old-fashioned or deep learning or a combination?
- 75 - GUI Widget Detection and Intent Generation via Image Understanding

### Datasets
- RICO -> 72 and 74 : This dataset (android only) is pretty much crap, so much preprocessing is required to make it work, and other authors who used it did not publish their preprocessed dataset 
- Image2emmet: very small dataset, both for web and android
- [The ReDraw Dataset](https://zenodo.org/record/2530277#.ZAQ5mXbMJ3g)
- [VINS Dataset](https://github.com/sbunian/VINS)


#### Report
- cleaned the data and transform xml to txt (yolo5_format folder) -> See data_cleaning file
- see readme.md in yolo directory

## Metrics
- [https://medium.com/@vijayshankerdubey550/evaluation-metrics-for-object-detection-algorithms-b0d6489879f3](https://medium.com/@vijayshankerdubey550/evaluation-metrics-for-object-detection-algorithms-b0d6489879f3)

## Citation
```
@misc{daneshvar2024guielementdetectionusing,
      title={GUI Element Detection Using SOTA YOLO Deep Learning Models}, 
      author={Seyed Shayan Daneshvar and Shaowei Wang},
      year={2024},
      eprint={2408.03507},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.03507}, 
}
```
