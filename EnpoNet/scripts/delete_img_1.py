import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 设置相似度阈值
threshold = 0.95

# 遍历数据集中的所有图片
image_folder = 'path_to_image_folder'
image_files = os.listdir(image_folder)

# 定义处理函数
def process_image(image1, image_file1, image2, image_file2):
    ssim = cv2.SSIM(image1, image2)
    if ssim > threshold:
        os.remove(os.path.join(image_folder, image_file2))
        print(f"Deleted {image_file2} due to similarity with {image_file1}")

# 创建线程池
with ThreadPoolExecutor() as executor:
    for i, image_file1 in enumerate(image_files):
        image1 = cv2.imread(os.path.join(image_folder, image_file1))
        for j in range(i+1, len(image_files)):
            image_file2 = image_files[j]
            image2 = cv2.imread(os.path.join(image_folder, image_file2))
            executor.submit(process_image, image1, image_file1, image2, image_file2)
