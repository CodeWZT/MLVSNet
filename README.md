# PyTorch code for “MLVSNet: Multi-level Voting Siamese Network for 3D Visual Tracking”.

## Introduction

This repository is released for MLVSNet in our [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_MLVSNet_Multi-Level_Voting_Siamese_Network_for_3D_Visual_Tracking_ICCV_2021_paper.pdf). Here we include our MLVSNet model (PyTorch) and code for data preparation, training and testing on KITTI tracking dataset.
![teaser](https://github.com/CodeWZT/MLVSNet/blob/main/temp/framework.png)
## Preliminary

* Install ``python 3.6``.

* Install dependencies.
```
    pip install -r requirements.txt
```

* Build `_ext` module.
```
    python setup.py build_ext --inplace
```

* Download the dataset from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).

	Download [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) in the dataset and place them under the same parent folder.

## Evaluation

Train a new MLVSNet on KITTI data:
```
python train_tracking.py --data_dir=<kitti data path>
```

Test a new MLVSNet model on KITTI data:
```
python test_tracking.py --data_dir=<kitti data path>
```

Please refer to the code for setting of other optional arguments, including data split, training and testing parameters, etc.

## Acknowledgements
This code largely benefits from excellent works [P2B](https://github.com/HaozheQi/P2B), please also consider cite P2B if you use this code.
They help and inspire this work.

## Citation
```
@InProceedings{Wang_2021_ICCV,
    author    = {Wang, Zhoutao and Xie, Qian and Lai, Yu-Kun and Wu, Jing and Long, Kun and Wang, Jun},
    title     = {MLVSNet: Multi-Level Voting Siamese Network for 3D Visual Tracking},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {3101-3110}
}
```
