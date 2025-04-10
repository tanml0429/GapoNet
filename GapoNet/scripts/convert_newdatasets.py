import os
from pathlib import Path
import damei as dm
import cv2
import shutil
import numpy as np

def make_dir(tp, trte):
    """创建文件夹"""
    if not os.path.exists(tp):
        os.makedirs(tp, exist_ok=True)  # 创建目录
    os.makedirs(f'{tp}/images/{trte}', exist_ok=True)
    os.makedirs(f'{tp}/labels/{trte}', exist_ok=True)

    # 清空文件夹
    for root, dirs, files in os.walk(f'{tp}/images/{trte}'):
        for file in files:
            os.remove(os.path.join(root, file))

    # 清空文件夹
    for root, dirs, files in os.walk(f'{tp}/labels/{trte}'):
        for file in files:
            os.remove(os.path.join(root, file))

def rename_files(imgs_path):
    # 获取路径
    # current_dir = os.getcwd()
    img_list = []
    mask_list = []
    # 遍历当前文件夹中的所有文件
    for root, dirs, files in os.walk(imgs_path):
        for file in files:
            if Path(file).suffix in ['.jpg', '.png']:
                file_path = f'{root}/{file}'
                # print(file_path)
        # 构造原始文件名和新文件名
                file_path = Path(file_path).parent
                old_filename = os.path.join(file_path, file)
                new_filename = os.path.join(file_path, str(file_path).split('/')[-2] + '_' + file)
                # 如果是文件，就重命名
                if (str(file_path).split('/')[-2] not in file) and (os.path.isfile(old_filename)):
                    os.rename(old_filename, new_filename)   
                    if 'images' in old_filename:
                        img_list.append(new_filename)
                    elif 'masks' in old_filename:
                        mask_list.append(new_filename)

                elif (str(file_path).split('/')[-2] in file) and (os.path.isfile(old_filename)):
                    if 'images' in old_filename:
                        img_list.append(old_filename)
                    elif 'masks' in old_filename:
                        mask_list.append(old_filename)
        
            
    return img_list, mask_list

def mask2bbox(img_path):
        """根据mask获取bbox，注意需要处理红绿蓝等情况"""
        mask = cv2.imread(img_path)
        try:
            h, w, c = mask.shape
        except:
            h, w = mask.shape
            c = 1
        bboxes = []
        mask_white = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_white = cv2.inRange(mask, (200, 200, 200), (255, 255, 255))
        contours_white, hierarchy_white = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_white:
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = [x, y, x + w, y + h]

            
            bboxes.append(bbox)

        return bboxes

def xyxy2xywh(img_path, bbox):
        """将bbox从xyxy格式转换为yolo的xywh格式"""
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        y = np.zeros_like(bbox)

        y[0] = (bbox[0] + bbox[2]) / 2 / w
        y[1] = (bbox[1] + bbox[3]) / 2 / h
        y[2] = (bbox[2] - bbox[0]) / w
        y[3] = (bbox[3] - bbox[1]) / h
        return y

# 保存bbox
def save_bbox(img_path, bboxes, labels_path, filename):
    """保存bbox到txt文件"""
    # h, w, c = img.shape
    # bbox = dm.general.xywh2xyxy(bbox)
    # bbox[0] = bbox[0] / w
    # bbox[1] = bbox[1] / h
    # bbox[2] = bbox[2] / w
    # bbox[3] = bbox[3] / h  # in fraction
    for bbox in bboxes:
            bbox = [str(x).replace('\n', '') for x in bbox]
            bbox = [float(x) if x !='' else bbox for x in bbox]
            bbox = xyxy2xywh(img_path, bbox)
            if bbox[2] < 0.02 or bbox[3] < 0.02:
                continue
            bbox_str = f'{bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}'
            save_path = os.path.join(labels_path, filename.replace('.png', '.txt'))
            with open(save_path, 'a') as f:
                f.write(f'0 {bbox_str}\n') # 0表示类别



if __name__ == '__main__':
    # 1. 创建文件夹
    tp = '/data/tml/newdata'
    trte = 'train'
    make_dir(tp, trte)
    trte = 'test'
    make_dir(tp, trte)


    # 3. train保存图片及bbox
    for trte in ['train']:
        imgs_path = f'/data/tml/newdataset/TrainDataset/image'
        #imgs_path = f'/data/tml/newdata/test/image'
        for filename in os.listdir(imgs_path):
            img_path = os.path.join(imgs_path, filename)
            anno_path = img_path.replace('image', 'masks')
            # 复制图片到目标文件夹
            shutil.copy(img_path, f'{tp}/images/{trte}')
            labels_path = f'{tp}/labels/{trte}'
            bbox = mask2bbox(anno_path)
            save_bbox(anno_path, bbox, labels_path, filename)

    # 2. test重命名文件,保存图片及bbox
    
    imgs_path = '/data/tml/newdataset/TestDataset'
    img_list, mask_list = rename_files(imgs_path)
    for trte in ['test']:
        for i in img_list:
            img_path = i
            # 复制图片到目标文件夹
            shutil.copy(img_path, f'{tp}/images/{trte}')
        for i in mask_list:
            anno_path = i
            labels_path = f'{tp}/labels/{trte}'
            bbox = mask2bbox(anno_path)
            filename = anno_path.split('/')[-1]
            save_bbox(anno_path, bbox, labels_path, filename)

    

    

  
    
    

