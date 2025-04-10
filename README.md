[![Stars](https://img.shields.io/github/stars/tanml0429/GapoNet)](
https://github.com/tanml0429/GapoNet)
[![Open issue](https://img.shields.io/github/issues/tanml0429/GapoNet)](
https://github.com/tanml0429/GapoNet/issues)
[![Datasets](https://img.shields.io/static/v1?label=Download&message=source_code&color=orange)](
https://github.com/tanml0429/GapoNet/archive/refs/heads/main.zip)

#### English | [简体中文](https://github.com/tanml0429/GapoNet/blob/main/docs/README_zh_cn.md)

Please **star this project** in the upper right corner and **cite this paper** blow 
if this project helps you. 

# GapoNet

This repository is the source codes for the paper 
"Optimizing Gastrointestinal Polyp Recognition through Low-Redundancy Dataset".

GapoNet (Gastrointestinal Polyp Detection Network) is proposed to detect gastrointestinal polyps for endoscope images.

![GA](https://github.com/tanml0429/GapoNet/blob/main/docs/GA.jpg)

Graphical abstract.


# Contributions

1. A pretrained high-dimensional feature-driven cosine similarity deduplication technique was proposed to construct GapoSet, an AI dataset for gastrointestinal polyp recognition, characterized by low similarity and the great diversity.

2. Detection was conducted using the improved YOLO11 network (with CA and MHSA attention mechanisms), and state-of-the-art performance was achieved. 


# Installation
Get GapoNet code and configure the environment, please check out [docs/INSTALL.md](https://github.com/tanml0429/GapoNet/blob/master/docs/INSTALL.md)

# Datasets and trained weights
Download datasets and trained weights, please check out [docs/DATASETS.md](https://github.com/tanml0429/GapoNet/blob/master/docs/DATASETS.md)

# Quick Start

## Validation
```
python val.py
```
The main optional arguments:
```
--model "xx.pt"
--data  "xx.yaml"
--device "0, 1"  # cpu or gpu id
--imgsz 640 
--batch 32
--base_save_dir "xx" 

```


## Train

Once you get the LymoNet code, configure the environment and download the dataset, just type:
```
python train.py 
```
The training results and weights will be saved in runs/detect/directory.

The main optional arguments:
```
--model "x.yaml"
--data  "xx.yaml"
--device "0, 1"  # cpu or gpu id
--imgsz 640 
--batch 32 
--epochs 300 
```


# Contributors

GapoNet is authored by Menglu Tan, Zhengde Zhang, Ao Wang, Zijin Zeng, and Lin Feng.

Currently, it is maintained by Menglu Tan (tanml0429@gmail.com).

# Acknowledgement

This work was supported by the Beijing Municipal Fund for Distinguished Young Scholars (Grand No. JQ22022), National Key R&D Program of China (Grant No. 2022YFF1502000).

We are very grateful to the [ultralytics](https://github.com/ultralytics/ultralytics) project for the benchmark detection algorithm.



# Citation
```

```


# License
GapoNet is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at tanml0429@gmail.com. We will send the detail agreement to you.