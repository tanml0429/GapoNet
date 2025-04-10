# Installation

This document describes how to get EnpoNet source code and configure the running environment.

## Requirements
+ Python 3.10+
+ Pytorch 2.1+
+ Linux or MacOS

## Get EnpoNet Code

##### use git:
```python
git clone https://github.com/tanml0429/EnpoNet.git
cd EnpoNet
```
##### from github web:
+ Browse the [EnpoNet] repository
+ Click "Code"-"Download ZIP"
```python
unzip EnpoNet-master.zip
cd EnpoNet_master
```


## Configure Environment
[Anaconda](https://www.anaconda.com) is highly recommanded.

Haven't Anaconda been installed yet? Download anaconda installer [here](https://www.anaconda.com/products/individual#Downloads) and install it:
```python
chmod +x ./Anaconda3-2020.11-Linux-.sh
./Anaconda3-2020.11-Linux-.sh  # install
which conda  # verify installation
```

After having a CONDA, directly import the fully configured environment:
```python
conda env creat -f conda_lymph_env.yaml
```

or Creating a environment from sratch:
```python
conda create --name lymph python=3.7  # create a env named as "lymph"
conda activate lymph  # activate the env
which pip  # verify pip 
pip install -r requirements.txt  # install packages use pip
# or use conda to install package
conda install <PACKAGE>
```