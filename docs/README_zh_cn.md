
#### [English](https://github.com/tanml0429/GapoNet) | 简体中文

如果本项目对你有帮助，请点击项目右上角star支持一下和引用下面的论文。

# LymoNet

这个项目是论文的《基于低冗余数据集的胃肠道息肉识别优化》的代码、数据集和教程。

 GapoNet(胃肠道息肉检测网络)被提出用于从胃肠道内窥镜图像中检测息肉。

![GA](https://github.com/tanml0429/GapoNet/blob/main/docs/GA.jpg)

图形摘要


# 贡献Contributions

1. 提出了一种预训练的高维特征驱动余弦相似度重复数据删除技术，用于构建低相似度、大多样性的胃肠道息肉识别人工智能数据集GapoSet。

2. 使用改进的YOLO11网络（加入CA和MHSA注意机制）进行检测，并实现了最先进的性能。


# 安装Installation
安装LymoNet代码并配置环境，请查看：
[docs/INSTALL.md](https://github.com/tanml0429/GapoNet/blob/master/docs/INSTALL.md)

# 验证集和权重Validation data and trained weights
下载验证集和训练好的权重，请查看：
[docs/DATASETS.md](https://github.com/tanml0429/GapoNet/blob/master/docs/DATASETS.md)

# 快速开始Quick Start
## 训练Train

安装GapoNet代码，配置环境和下载数据集后，输入代码训练：
```
python train.py 
```
训练结果和权重将保存在 base_save_dir 目录中。

主要的可选参数：
```
--model "x.yaml"
--data  "xx.yaml"
--device "0, 1"  # cpu or gpu id
--imgsz 640 
--batch 32 
--epochs 300
--base_save_dir "xx" 
```

## 验证val

```
python val.py
```

主要的可选参数:
```
--model "xx.pt"
--data  "xx.yaml"
--device "0, 1"  # cpu or gpu id
--imgsz 640 
--batch 32
```









# 贡献者Contributors
GapoNet的作者是: Menglu Tan, Zhengde Zhang, Ao Wang, Zijin Zeng, and Lin Feng

目前，GapoNet由Menglu Tan (tanml0429@gmail.com)负责维护。

如果您有任何问题，请随时与我们联系。



# 致谢Acknowledgement

本项工作得到了以下资助：北京市杰出青年学者基金（项目编号：JQ22022）以及国家重点研发计划（项目编号：2022YFF1502000）。

我们非常感谢
[ultralytics](https://github.com/ultralytics/ultralytics)
项目提供的目标检测算法基准。



如果对您有帮助，请为点击项目右上角的star支持一下或引用论文。

# 引用Citation



# 许可License
GapoNet可免费用于非商业用途，并可在这些条件下重新分发。 如需商业咨询，请发送电子邮件至tanml0429@gmail.com，我们会将详细协议发送给您。