import os
import cv2
import albumentations as A
from tqdm import tqdm
import shutil
# from albumentations.augmentations.bbox_utils import convert_bbox_from_albumentations, convert_bbox_to_albumentations

# 设置原始图片文件夹路径和增强后图片文件夹路径
original_image_folder = '/home/tml/datasets/enpo_dataset/enpo_10341_t/images'  # 原始图片文件夹路径
original_label_folder = '/home/tml/datasets/enpo_dataset/enpo_10341_t/labels'  # 原始标注文件夹路径
augmented_image_folder = '/home/tml/datasets/enpo_dataset/aug_61730/images'  # 增强后图片文件夹路径
augmented_label_folder = '/home/tml/datasets/enpo_dataset/aug_61730/labels'  # 增强后标注文件夹路径
# # 创建增强后图片文件夹
# if not os.path.exists(augmented_image_folder):
#     os.makedirs(augmented_image_folder)

# # 创建增强后标签文件夹
# if not os.path.exists(augmented_label_folder):
#     os.makedirs(augmented_label_folder)

# # 定义 Albumentations 的增强操作
# transform = A.Compose([
#     A.HorizontalFlip(p=0.5), # 水平翻转概率为0.5
#     A.VerticalFlip(p=0.5), # 垂直翻转概率为0.5
#     A.RandomRotate90(p=0.5), # 随机旋转90度概率为0.5
#     A.RandomBrightnessContrast(p=0.2), # 随机亮度和对比度
#     A.Blur(p=0.2), # 模糊概率为0.2
#     A.ShiftScaleRotate(p=0.5), # 平移、缩放和旋转概率为0.5
#     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3), # 色调、饱和度和亮度
#     A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-10, 10), shear=(-5, 5), p=0.3), # 仿射变换
#     A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), # 高斯噪声
#     # A.RandomCrop(height=256, width=256, p=0.5), # 随机裁剪
#     A.ChannelShuffle(p=0.2), # 通道混洗
#     # A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.3), # 随机遮挡
#     A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3), # 光学失真
# ], bbox_params=A.BboxParams(format='yolo', label_fields=[]))

# # 对每张原始图片进行增强并保存
# for image_name in os.listdir(original_image_folder):
#     image_path = os.path.join(original_image_folder, image_name)
#     _, ext = os.path.splitext(image_name)
#     if ext.lower() in ['.jpg', '.jpeg', '.png']:
#         image = cv2.imread(image_path)
    
#         # 读取 YOLO 格式的标注信息
#         label_name = image_name.replace(ext, '.txt')
#         label_path = os.path.join(original_label_folder, label_name)
#         with open(label_path, 'r') as file:
#             annotations = file.readlines()
    
#         # 将标注信息转换为 Albumentations 格式
#         bboxes = [list(map(float, annotation.strip().split()[1:])) for annotation in annotations]
    
#     # 生成n倍的增强照片,加进度条
#     for i in range(7):
#         augmented = transform(image=image, bboxes=bboxes)
        
#         # 保存增强后的图片和标注信息
#         augmented_image = augmented["image"]
#         augmented_bboxes = augmented["bboxes"]
#         augmented_image_path = os.path.join(augmented_image_folder, image_name.split('.')[0] + f'_aug_{i}{ext}')
#         cv2.imwrite(augmented_image_path, augmented_image)
        
#         # 将增强后的标注信息保存为 YOLO 格式
#         with open(os.path.join(augmented_label_folder, image_name.split('.')[0] + f'_aug_{i}.txt'), 'w') as file:
#             for bbox in augmented_bboxes:
#                 file.write('0 ' + ' '.join(map(str, bbox)) + '\n')

# print("完成数据增强操作")

# # 拷贝原始图片和标注信息到增强后文件夹
# for image_name in os.listdir(original_image_folder):
#     image_path = os.path.join(original_image_folder, image_name)
#     label_name = image_name.replace(os.path.splitext(image_name)[1], '.txt')
#     label_path = os.path.join(original_label_folder, label_name)
    
#     shutil.copy(image_path, os.path.join(augmented_image_folder, image_name))
#     shutil.copy(label_path, os.path.join(augmented_label_folder, label_name))

# # 删除尾号为8的图片和标注信息
# for image_name in os.listdir(augmented_image_folder):
#     if image_name.endswith('_aug_6.jpg') or image_name.endswith('_aug_6.png') or image_name.endswith('_aug_6.jpeg'):
#         os.remove(os.path.join(augmented_image_folder, image_name))
#         os.remove(os.path.join(augmented_label_folder, image_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')))

# # 删除尾号为7的随机一半的图片和标注信息
# import random

# # 获取尾号为7的图片列表
# images_to_delete = [image_name for image_name in os.listdir(augmented_image_folder) if image_name.endswith('_aug_7.jpg') or image_name.endswith('_aug_7.png')] or image_name.endswith('_aug_7.jpeg')

# # 随机选择一半图片删除
# random.shuffle(images_to_delete)
# images_to_delete = images_to_delete[:len(images_to_delete)//2]

# # 删除选中的图片和对应的标注信息
# for image_name in images_to_delete:
#     os.remove(os.path.join(augmented_image_folder, image_name))
#     os.remove(os.path.join(augmented_label_folder, image_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))

# print("删除尾号为7的随机一半的图片和标注信息完成")

# 随机删除一万张图片和标注
import random
images_to_delete = random.sample(os.listdir(augmented_image_folder), 3000)
for image_name in images_to_delete:
    os.remove(os.path.join(augmented_image_folder, image_name))
    os.remove(os.path.join(augmented_label_folder, image_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')))
print("删除一万张图片和标注完成")
    



#打印图片数量
print(f"原始图片数量: {len(os.listdir(original_image_folder))}")
print(f"增强后图片数量: {len(os.listdir(augmented_image_folder))}")







