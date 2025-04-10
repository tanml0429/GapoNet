# Dataset

The GapoSet dataset contains 10341 images for train, 1293 images for validation and 1293 images for test.

Here, we provide validation data.

## Download data
+ Download from [Baidu Pan](https://pan.baidu.com/s/199rTvq6B7vl1x7R47g-dCA), passwd: **v9bn**

+ Download from [Google Drive](https://drive.google.com/file/d/1dDAWH74Z8s6RT83HQitk8LjVewnHSBPk/view?usp=drive_link).

**Notes**

Please remove proxy if download failed.


The file structure format is YOLO11 likes:
```
|--data_v8_format
--|--images # images for train, val and test
   --|--train  
      -- xxx.jpg
      -- ...
   --|--val 
      -- xxx.jpg
      -- ...
   --|--test  
      -- xxx.jpg
      -- ...   
--|--labels  # corresponding .txt labels
   --|--train  
      -- xxx.txt
      -- ...
   --|--val  
      -- xxx.txt
      -- ...
   --|--test  
      -- xxx.txt
      -- ...
```
Each .txt file contains annotations in the format of CLS XC YC W H in each line. 

CLS(Classes): 0 polyp

XC YC W H in terms of percentage.

# Trained weight

The improved SOTA trained weight is provided.
+ Download from [Baidu Pan](https://pan.baidu.com/s/1AYbcqvXVNPrWHS3TMsq5QQ), passwd: **lq6p**

+ Download from [Google Drive](https://drive.google.com/file/d/1iShl8K0VgYdjjEuzUU8kH5K_MhcD3sRN/view?usp=drive_link).


