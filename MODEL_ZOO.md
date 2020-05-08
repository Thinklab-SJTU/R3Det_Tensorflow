# R<sup>3</sup>Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object

## Performance
### Baseline
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Anchor | Reg. Loss| Angle Range | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) (baseline) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 53.17 | - | H | smooth L1 | 90 | 1x | No | 8X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_v3.py](./libs/configs/DOTA1.0/baseline/cfgs_res50_dota_v3.py) |    
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) (baseline) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 63.18 | [model](https://drive.google.com/file/d/18Z3NWhL4gQB5yJLCXBcHBnK-6BPle3m1/view?usp=sharing) | H | smooth L1 | 90 | 1x | No |**1X** GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_v4.py](./libs/configs/DOTA1.0/baseline/cfgs_res50_dota_v4.py) |     
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) (baseline) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 62.79 | - | H | smooth L1 | 90 | **2x** | No | 8X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_v8.py](./libs/configs/DOTA1.0/baseline/cfgs_res50_dota_v8.py) |     
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) (baseline) | **ResNet101_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 64.73 | [model](https://drive.google.com/file/d/16XCbS9T-tTr7ySa-qlyc8TuufAxJTvD4/view?usp=sharing) | H | smooth L1 | 90 | 1x | No | 1X GeForce RTX 2080 Ti | 1 | [cfgs_res101_dota_v9.py](./libs/configs/DOTA1.0/baseline/cfgs_res101_dota_v9.py) |   
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) (baseline) | **ResNet152_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 66.97 | [model](https://drive.google.com/file/d/1OdexYmEGH8hyucbdKZinGqvRNFveZ1AQ/view?usp=sharing) | H | smooth L1 | 90 | 1x | No | 1X GeForce RTX 2080 Ti | 1 | [cfgs_res152_dota_v12.py](./libs/configs/DOTA1.0/baseline/cfgs_res101_dota_v12.py) |
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 65.11 | [model](https://drive.google.com/file/d/1vnJd4FEvy61yvcYqhZJgfnxJ3fByULOR/view?usp=sharing) | H | smooth L1 + **atan(theta)** | 90 | 1x | No | 1X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_v16.py](./libs/configs/DOTA1.0/baseline/cfgs_res101_dota_v16.py) |     
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 64.10 | [model](https://drive.google.com/file/d/1SgiDME_gHzKrFxoZSjS9E-_QbGiBr9lV/view?usp=sharing) | H | smooth L1 | **180** | 1x | No | 1X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_v15.py](./libs/configs/DOTA1.0/baseline/cfgs_res101_dota_v15.py) |     
|  |  |  |  |  |  |  |  |  |  |  |  |  |
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) (baseline) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 62.76 | [model](https://drive.google.com/file/d/1n0O6qLJjdDewb_9FDgsGkISevL7SLD8_/view?usp=sharing) | R | smooth L1 | 90 | 1x | No | 1X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_v1.py](./libs/configs/DOTA1.0/baseline/cfgs_res50_dota_v1.py) |
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) (baseline) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 62.25 | - | R | smooth L1 | 90 | **2x** | No | **8X** GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_v10.py](./libs/configs/DOTA1.0/baseline/cfgs_res50_dota_v10.py) |
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 68.65 | [model](https://drive.google.com/file/d/17RLZK0CwIgqtCAnifa0huWCa3EAaTw_l/view?usp=sharing) | R | [**iou-smooth L1 [-ln(x)]**](https://arxiv.org/abs/1811.07126) | 90 | 1x | No | 1X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_v5.py](./libs/configs/DOTA1.0/baseline/cfgs_res50_dota_v5.py) |    

### R<sup>3</sup>Det
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Anchor | Reg. Loss| Angle Range | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| [R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 66.31 | [model](https://drive.google.com/file/d/1cBKxcePQFIv3yKQTOVw598nb-IwUXJV_/view?usp=sharing)  | H + R | smooth L1 | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_v1.py](./libs/configs/DOTA1.0/r3det/cfgs_res50_dota_r3det_v1.py) |
| [R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 67.29 (67.66) | [model](https://drive.google.com/file/d/1RfnLyNgy5pwVuCvOmGao0ytC5jfSRQjx/view?usp=sharing) | H + R | smooth L1 | 90 | 2x | No | 2X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_v2.py](./libs/configs/DOTA1.0/r3det/cfgs_res50_dota_r3det_v2.py) |
| [R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612) | **ResNet101_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 71.69 | -  | H + R | smooth L1 | 90 | 3x | Yes | 8X GeForce RTX 2080 Ti | 1 | - |
| [R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612) | **ResNet152_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 72.81 | -  | H + R | smooth L1 | 90 | **4x** | Yes | 8X GeForce RTX 2080 Ti | 1 | - |
| [R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612) | **ResNet152_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 73.74 | -  | H + R | smooth L1 | 90 | **4x** | Yes | 8X GeForce RTX 2080 Ti | 1 | - |

### Anchor Free
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Anchor | Anchor Scale |  Anchor Ratio | Positive Threshold | Negative Threshold | Reg. Loss| Angle Range | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| [R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 67.66 | [model](https://drive.google.com/file/d/1RfnLyNgy5pwVuCvOmGao0ytC5jfSRQjx/view?usp=sharing) | H + R | [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)] | [1, 1 / 2, 2., 1 / 3., 3., 5., 1 / 5.] | [0.5, 0.6, 0.7] | [0.4, 0.5, 0.6] | smooth L1 | 90 | 2x | No | 2X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_v2.py](./libs/configs/DOTA1.0/r3det/cfgs_res50_dota_r3det_v2.py) |
| [R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 65.89 | - | H + R | [1.] | [1.] | [0.5, 0.6, 0.7] | [0.4, 0.5, 0.6] | smooth L1 | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_v9.py](./libs/configs/DOTA1.0/r3det/cfgs_res50_dota_r3det_v9.py) |
| [R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 67.11 | - | H + R | [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)] | [1.] | [0.35, 0.5, 0.6] | [0.25, 0.4, 0.5] | smooth L1 | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_v5.py](./libs/configs/DOTA1.0/r3det/cfgs_res50_dota_r3det_v5.py) |
| [R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 67.32 | - | H + R | [1.] | [1.] | [0.35, 0.5, 0.6] | [0.25, 0.4, 0.5] | smooth L1 | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_v7.py](./libs/configs/DOTA1.0/r3det/cfgs_res50_dota_r3det_v7.py) |
| [R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 69.50 | - | H + R | [1.] | [1.] | [0.35, 0.5, 0.6] | [0.25, 0.4, 0.5] | [**iou-smooth L1 [1-exp(1-x)]**](https://arxiv.org/abs/1811.07126) | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_v12.py](./libs/configs/DOTA1.0/r3det/cfgs_res50_dota_r3det_v12.py) |

### R<sup>3</sup>Det++
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | InLD | Anchor | Reg. Loss| Angle Range | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| **[R<sup>3</sup>Det++](https://yangxue0827.github.io/SCRDet++.html)** | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 69.07 | - | {4,4,3,2,2} | H + R | smooth L1 | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_plusplus_v2.py](./libs/configs/DOTA1.0/r3det_plusplus/cfgs_res50_dota_r3det_plusplus_v2.py) |
| [R<sup>3</sup>Det++](https://yangxue0827.github.io/SCRDet++.html) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 69.81 | [model](https://drive.google.com/file/d/12_7-7ihl5Nozvja6aQjLEzxY8RUy0px5/view?usp=sharing) | **{1,1,1,1,1}** | H + R | smooth L1 | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_plusplus_v3.py](./libs/configs/DOTA1.0/r3det_plusplus/cfgs_res50_dota_r3det_plusplus_v3.py) |
| [R<sup>3</sup>Det++](https://yangxue0827.github.io/SCRDet++.html) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 70.08 | - | {1,1,1,1,1} | H + R | [**iou-smooth L1 [1-exp(1-x)]**](https://arxiv.org/abs/1811.07126) | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_plusplus_v9.py](./libs/configs/DOTA1.0/r3det_plusplus/cfgs_res50_dota_r3det_plusplus_v9.py) |
| [R<sup>3</sup>Det++](https://yangxue0827.github.io/SCRDet++.html) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 68.12 | - | {1,1,1,1,1} + **binary** | H + R | smooth L1 | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_plusplus_v7.py](./libs/configs/DOTA1.0/r3det_plusplus/cfgs_res50_dota_r3det_plusplus_v7.py) |
| [R<sup>3</sup>Det++](https://yangxue0827.github.io/SCRDet++.html) | **ResNet101_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 69.19 | - | {1,1,1,1,1} | H + R | smooth L1 | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res101_dota_r3det_plusplus_v4.py](./libs/configs/DOTA1.0/r3det_plusplus/cfgs_res50_dota_r3det_plusplus_v4.py) |
| [R<sup>3</sup>Det++](https://yangxue0827.github.io/SCRDet++.html) | ResNet101_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 72.98 | - | {1,1,1,1,1} | H + R | smooth L1 | 90 | 3x | Yes | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res101_dota_r3det_plusplus_v5.py](./libs/configs/DOTA1.0/r3det_plusplus/cfgs_res50_dota_r3det_plusplus_v5.py) |
| [R<sup>3</sup>Det++](https://yangxue0827.github.io/SCRDet++.html) | **ResNet152_v1d** 600->800 | DOTA1.0 trainval | DOTA1.0 test | 74.41 | - | {4,4,3,2,2} | H + R | smooth L1 | 90 | 4x | Yes | 8X GeForce RTX 2080 Ti | 1 | - |
| [R<sup>3</sup>Det++](https://yangxue0827.github.io/SCRDet++.html) | ResNet152_v1d **MS** | DOTA1.0 trainval | DOTA1.0 test | 76.56 | [model](https://drive.google.com/file/d/1DTEwh1Uyj14PgCjGFZW4jOdWdMw7GJQf/view?usp=sharing)  | {4,4,3,2,2} | H + R + more | smooth L1 | 90 | 6x | Yes | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res152_dota_r3det_plusplus_v1.py](./libs/configs/DOTA1.0/r3det_plusplus/cfgs_res50_dota_r3det_plusplus_v1.py) |

### IoU-Smooth L1 Loss
| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Anchor | Reg. Loss| Angle Range | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) (baseline) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 62.25 | - | R | smooth L1 | 90 | **2x** | No | **8X** GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_v10.py](./libs/configs/DOTA1.0/baseline/cfgs_res50_dota_v10.py) |
| [RetinaNet](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 68.65 | [model](https://drive.google.com/file/d/17RLZK0CwIgqtCAnifa0huWCa3EAaTw_l/view?usp=sharing) | R | [**iou-smooth L1 [-ln(x)]**](https://arxiv.org/abs/1811.07126) | 90 | 1x | No | 1X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_v5.py](./libs/configs/DOTA1.0/baseline/cfgs_res50_dota_v5.py) |    

| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | Anchor | Anchor Scale |  Anchor Ratio | Positive Threshold | Negative Threshold | Reg. Loss| Angle Range | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| [R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 67.32 | - | H + R | [1.] | [1.] | [0.35, 0.5, 0.6] | [0.25, 0.4, 0.5] | smooth L1 | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_v7.py](./libs/configs/DOTA1.0/r3det/cfgs_res50_dota_r3det_v7.py) |
| [R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 69.50 | - | H + R | [1.] | [1.] | [0.35, 0.5, 0.6] | [0.25, 0.4, 0.5] | [**iou-smooth L1 [1-exp(1-x)]**](https://arxiv.org/abs/1811.07126) | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_v12.py](./libs/configs/DOTA1.0/r3det/cfgs_res50_dota_r3det_v12.py) |

| Model |    Backbone    |    Training data    |    Val data    |    mAP   | Model Link | InLD | Anchor | Reg. Loss| Angle Range | lr schd | Data Augmentation | GPU | Image/GPU | Configs |      
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|    
| [R<sup>3</sup>Det++](https://yangxue0827.github.io/SCRDet++.html) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 69.81 | [model](https://drive.google.com/file/d/12_7-7ihl5Nozvja6aQjLEzxY8RUy0px5/view?usp=sharing) | **{1,1,1,1,1}** | H + R | smooth L1 | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_plusplus_v3.py](./libs/configs/DOTA1.0/r3det_plusplus/cfgs_res50_dota_r3det_plusplus_v3.py) |
| [R<sup>3</sup>Det++](https://yangxue0827.github.io/SCRDet++.html) | ResNet50_v1d 600->800 | DOTA1.0 trainval | DOTA1.0 test | 70.08 | - | {1,1,1,1,1} | H + R | [**iou-smooth L1 [1-exp(1-x)]**](https://arxiv.org/abs/1811.07126) | 90 | 2x | No | 4X GeForce RTX 2080 Ti | 1 | [cfgs_res50_dota_r3det_plusplus_v9.py](./libs/configs/DOTA1.0/r3det_plusplus/cfgs_res50_dota_r3det_plusplus_v9.py) |

### EfficientNet
| Model |    Backbone    |    Training data    |    Val data    | Anchor | Anchor Scale |  Anchor Ratio | Positive Threshold | Negative Threshold | Reg. Loss| Angle Range | lr schd | Data Augmentation | GPU | Image/GPU |      
|:------------:|:------------:|:------------:|:---------:|:----------:|:-----------:|:----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|   
| [R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612) | **EfficientNet** 600->800 | DOTA1.0 trainval | DOTA1.0 test | H + R | [1.] | [1.] | [0.35, 0.5, 0.6] | [0.25, 0.4, 0.5] | [**iou-smooth L1 [1-exp(1-x)]**](https://arxiv.org/abs/1811.07126) | 90 | 2x | No | 2X GeForce RTX 2080 Ti | 1 |

We have provided a list of EfficientNet checkpoints for EfficientNet checkpoints:.

  * With baseline ResNet preprocessing, we achieve similar results to the original ICML paper.
  * With [AutoAugment](https://arxiv.org/abs/1805.09501) preprocessing, we achieve higher accuracy than the original ICML paper.
  * With [RandAugment](https://arxiv.org/abs/1909.13719) preprocessing, accuracy is further improved.
  * With [AdvProp](https://arxiv.org/abs/1911.09665), state-of-the-art results (w/o extra data) are achieved.
  * With [NoisyStudent](https://arxiv.org/abs/1911.04252), state-of-the-art results (w/ extra JFT-300M unlabeled data) are achieved.

|               |   B0    |  B1   |  B2    |  B3   |  B4   |  B5    | B6 | B7 | B8 | L2-475 | L2 |
|----------     |--------  | ------| ------|------ |------ |------ | --- | --- | --- | --- |--- |
| Baseline preprocessing | 60.29 ([cfgs](./libs/configs/DOTA1.0/r3det/cfgs_res50_dota_r3det_v14.py)) |   |  |  |  |  | | || | | |
| AutoAugment (AA) |    |  |  |  |  |  |   |  || | |
| RandAugment (RA) |  |  |  |  |  |  |  |  |  | | |
| AdvProp + AA |  |   |  |  |  |  |  |  | || | |
| NoisyStudent + RA |  |  |  |  | |  |  | |  | |  | 


[R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612): R<sup>3</sup>Det with two refinement stages      
**Some model results are slightly higher than in the paper due to retraining.**
