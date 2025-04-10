import os, sys
from pathlib import Path
import cv2
import numpy as np
import shutil
import damei as dm
import xml.etree.ElementTree as ET

class Convertor(object):

    def __init__(self, source_path, target_path) -> None:
        self.sp = source_path  
        self.tp = target_path  
        self.ratio = [0.8, 0.2]  # 训练集、测试集的比例

        self.force = False  # 是否强制覆盖, 如果为False，当目标路径存在时，不会覆盖

    def __call__(self):

        sp = self.sp
        tp = self.tp
        
        dataset_names = os.listdir(sp)  # 数据集名称
        # print(f'数据集名称：{dataset_names}')
        # 数据集路径
        for dataset_name in dataset_names:
            dataset_path = os.path.join(sp, dataset_name)
            # 1.统计数据集，获取所有图像的路径及其对应的标注路径
            img2anno_dict = self.statics_source(dataset_name)  # key: img_path, value: anno_path
            print(f'总计图像数目：{len(img2anno_dict)}')

            # 2.划分训练集、验证集和测试
            trte_paths = self.splite_dataset(img2anno_dict)

            # 3.处理数据集和保存
            for i, trte in enumerate(['train', 'test']):
                img_paths = trte_paths[i]
                self.process(f'{tp}/{dataset_name}', img2anno_dict, img_paths, dataset_path, trte)  

    def process(self, tp, img2anno_dict, img_paths, dataset_path, trte):
        
        # 创建保存路径
        if not os.path.exists(tp):
            os.makedirs(tp, exist_ok=True)  # 创建目录
        os.makedirs(f'{tp}/images/{trte}', exist_ok=True)
        os.makedirs(f'{tp}/labels/{trte}', exist_ok=True)

        # 清空文件夹
        # 为什么要清空？
        for root, dirs, files in os.walk(f'{tp}/images/{trte}'):
            for file in files:
                os.remove(os.path.join(root, file))

        # 清空文件夹
        for root, dirs, files in os.walk(f'{tp}/labels/{trte}'):
            for file in files:
                os.remove(os.path.join(root, file))

        static = {
            'n1': 0,
            'n2': 0,
        }
        for i, img_path in enumerate(img_paths):
            static['n1'] += 1
            anno_path = img2anno_dict[img_path]
            # print (anno_path)  
            if Path(anno_path).suffix in ['.jpg', '.jpeg', '.png']:
                # 读取标注
                # anno = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
                # bbox = self.mask2bbox(anno)
                # 标注为txt文件，读取txt文件


                # 读取彩色mask图像
                mask = cv2.imread(anno_path)
                bboxes = []

                # 将红色，绿色，白色物体转换为二进制mask图像
                mask_red = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask_red = cv2.inRange(mask, (0, 0, 200), (10, 10, 255))

                mask_green = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask_green = cv2.inRange(mask, (0, 200, 0), (10, 255, 10))

                mask_white = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask_white = cv2.inRange(mask, (200, 200, 200), (255, 255, 255))

                # 找到每个二进制mask图像中的轮廓
                contours_red, hierarchy_red = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_green, hierarchy_green = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_white, hierarchy_white = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 计算每个轮廓的bbox信息，并将其打印出来
                for cnt in contours_red:
                    x, y, w, h = cv2.boundingRect(cnt)
                    bbox = [x, y, x + w, y + h]
                    bboxes.append(bbox)
                    
                for cnt in contours_green:
                    x, y, w, h = cv2.boundingRect(cnt)
                    bbox = [x, y, x + w, y + h]
                    bboxes.append(bbox)

                for cnt in contours_white:
                    x, y, w, h = cv2.boundingRect(cnt)
                    bbox = [x, y, x + w, y + h]
                    bboxes.append(bbox)
                # print(bboxes)
            
            # txt转化为bbox
            if Path(anno_path).suffix in ['.txt']:
                bbox = []
                with open(anno_path, 'r') as f:
                    lines = f.readlines()
                bboxes = []
                if lines[0] == '0' or lines[0] == '0\n':
                    bbox = []
                    # print (f'图像{img_path}没有标注')
                else:
                    for line in lines[1:]:
                        # print(line)
                        bbox = line.split(' ')
                        bboxes.append(bbox)
            # xml转化为bbox
            if Path(anno_path).suffix in ['.xml']:
                bboxes = []
                tree = ET.parse(anno_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    xmin = int(obj.find('object').find('bndbox').find('xmin').text)
                    ymin = int(obj.find('object').find('bndbox').find('ymin').text)
                    xmax = int(obj.find('object').find('bndbox').find('xmax').text)
                    ymax = int(obj.find('object').find('bndbox').find('ymax').text)
                    bbox = [xmin, ymin, xmax, ymax]
                    bboxes.append(bbox)
            
            # 拷贝图像
            img_name = Path(img_path).name
            # 相对路径
            relative_path = img_path.replace(dataset_path, '')
            # print(relative_path)
            # 把/替换为_
            img_name = relative_path.replace('/', '_')
            #imgname去掉后缀
            img_item = img_name.split('.')[0]

            # 把无标注图片拷贝到unlabeled文件夹
            if bboxes == []:
                unlabeled_dir = f'{Path(tp).parent.parent}/mixed_polyp_unlabeled/{Path(tp).name}'
                if not os.path.exists(unlabeled_dir):
                    os.makedirs(unlabeled_dir)
                if os.path.exists(f'{unlabeled_dir}/{img_name}'):
                    if self.force:
                        shutil.copy(img_path, f'{unlabeled_dir}/{img_name}')
                    else:
                        continue
                else:
                    shutil.copy(img_path, f'{unlabeled_dir}/{img_name}')
                
            else:
                # 画图
                # self.plot_img_and_show(img_path, bboxes, )
                
                if os.path.exists(f'{tp}/images/{trte}/{img_name}'):  
                    if self.force:  
                        shutil.copy(img_path, f'{tp}/images/{trte}/{img_name}')
                    else:
                        continue
                else:
                    shutil.copy(img_path, f'{tp}/images/{trte}/{img_name}')

            # 保存标注
            # assert bboxes != [], f'图像{img_path}没有标注'

            for bbox in bboxes:
                #在列表中去除换行符
                # bbox = [str(x).replace('\n', '') for x in bbox]
                bbox = [float(x) for x in bbox if str(x) != '\n']
                #str转float
                # bbox = [float(x) if x !='' else bbox for x in bbox]
                # bbox = [float(x) for x in bbox]
                #转换为yolo格式
                bbox = self.xyxy2xywh(img_path, bbox)
                # bbox = self.xyxy2xyxy_fraction(img_path, bbox)
                # bbox = dm.general.xyxy2xywh(np.array(bbox))
                bbox_str = f'{bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}'
                # print(bbox_str)
            
            

                # if os.path.exists(f'{tp}/labels/{trte}/{img_item}.txt'):
                #     if self.force:
                #         with open(f'{tp}/labels/{trte}/{img_item}.txt', 'w') as f:
                #             f.write(f'0 {bbox_str}\n') # 0表示类别
                #     else:
                #             continue
                # else:
                with open(f'{tp}/labels/{trte}/{img_item}.txt', 'a') as f:
                    f.write(f'0 {bbox_str}\n') # 0表示类别

        
                       
            
            print(f'\r{i+1}/{len(img_paths)} {img_path}', end='')  # 打印进度
        print()


        # 保存类别
        classes = ['polyp']
        with open(f'{tp}/classes.txt', 'w') as f:
            for c in classes:
                f.write(f'{c}\n')

    # def mask2bbox(self, mask):
    #     """根据mask获取bbox，注意需要处理红绿蓝等情况"""
    #     try:
    #         h, w, c = mask.shape
    #     except:
    #         h, w = mask.shape
    #         c = 1
    #     bbox = dm.general.mask2bbox(mask)  # 注意，该函数仅能把纯白色的mask转换为bbox，需做修改
    #     bbox_xywh = dm.general.xyxy2xywh(bbox)  # in pixel
    #     bbox_xywh[0] = bbox_xywh[0] / w
    #     bbox_xywh[1] = bbox_xywh[1] / h
    #     bbox_xywh[2] = bbox_xywh[2] / w
    #     bbox_xywh[3] = bbox_xywh[3] / h  # in fraction
    #     return bbox_xywh

    def plot_img_and_show(self, img_path, bboxes, **kwargs):

        img = cv2.imread(img_path)
        for bbox in bboxes:
            img = dm.general.plot_one_box_trace_pose_status(
                bbox, img, color=(0, 255, 0), label='polyp', line_thickness=2)
        name = Path(img_path).name
        cv2.imwrite(f'/data/tml/mixed_polyp_show/{name}', img)

    # def xyxy2xyxy_fraction(self, img_path, bbox):
    #     img = cv2.imread(img_path)
    #     h, w, _ = img.shape
    #     bbox[0] = bbox[0] / w
    #     bbox[1] = bbox[1] / h
    #     bbox[2] = bbox[2] / w   
    #     bbox[3] = bbox[3] / h
    #     return bbox
    
    # def xyxy2xywh(self, x):
    #     y[0] = (x[0] + x[2]) / 2
    #     y[1] = (x[1] + x[3]) / 2
    #     y[2] = (x[2] - x[0]) / 2
    #     y[3] = (x[3] - x[1]) / 2


    def xyxy2xywh_bak(self, img_path, bbox):
        """将bbox从xyxy格式转换为yolo的xywh格式"""
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        bbox[0] = (bbox[0] + bbox[2]) / 2 / w
        bbox[1] = (bbox[1] + bbox[3]) / 2 / h
        bbox[2] = (bbox[2] - bbox[0]) / w
        bbox[3] = (bbox[3] - bbox[1]) / h
        return bbox

    def xyxy2xywh(self, img_path, bbox):
        """将bbox从xyxy格式转换为yolo的xywh格式"""
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        y = np.zeros_like(bbox)

        y[0] = (bbox[0] + bbox[2]) / 2 / w
        y[1] = (bbox[1] + bbox[3]) / 2 / h
        y[2] = (bbox[2] - bbox[0]) / w
        y[3] = (bbox[3] - bbox[1]) / h
        return y


    def splite_dataset(self, img2anno_dict):  # 划分训练集、测试集
        img_paths = list(img2anno_dict.keys())  # 图像路径list
        img_paths.sort()  # 排序
        # 打乱顺序
        import random
        random.shuffle(img_paths)
        # 划分训练、测试集
        train_num = int(len(img_paths) * self.ratio[0])
        test_num = len(img_paths) - train_num
        train_paths = img_paths[:train_num]  # 训练集路径list
        test_paths = img_paths[train_num:]   # 测试集路径list
        return [train_paths, test_paths]
    

    def statics_source(self, dataset_name):  # 统计数据集，获取所有图像的路径及其对应的标注路径
        
        img2anno_dict = dict()
        count = 0
        for root, dirs, files in os.walk(f'{self.sp}/{dataset_name}'):  # 遍历数据集
            for file in files:
                if Path(file).suffix in ['.jpg', '.jpeg', '.png']:
                    img_path = f'{root}/{file}'
                    if 'masks' in img_path:
                        continue
                    anno_path = self.search_anno_path(img_path)  # 根据图像路径搜索标注文件路径
                    # print (anno_path)
                    count += 1
                    # print(count, img_path)
                    img2anno_dict[img_path] = anno_path
                    # break
                else:
                    continue
        return img2anno_dict
    

                
    def search_anno_path(self, img_path):
        """根据图像路径搜索标注文件路径"""

        # img_path = Path(img_path)  # 绝对路径
        # rel_path = str(Path(img_path).relative_to(self.sp))
        # print(rel_path)

        # 1.把路径中images替换为masks
        anno_path = img_path.replace('images', 'masks')
        # print(anno_path)
        if 'images' in img_path and os.path.exists(anno_path):
            return anno_path
        
        # 2.其他的判断方式，找到后return
        # 标注后缀为xml
        # PolypsSet数据集
        if 'PolypsSet' in img_path:
            anno_path = img_path.replace('Image', 'Annotation')
            suffix = Path(anno_path).suffix
            anno_path = anno_path.replace(suffix, '.xml')
            if 'Image' in img_path and os.path.exists(anno_path):
                return anno_path
       # PLOS数据集
        if 'PLOS' in img_path:
            anno_path = img_path.replace('TrainValImages', 'Annotations')
            suffix = Path(anno_path).suffix
            anno_path = anno_path.replace(suffix, '.xml')
            if 'TrainValImages' in img_path and os.path.exists(anno_path):
                return anno_path
        
        # 标注后缀为txt
        # LDTrainValid数据集
        if 'LDTrainValid' in img_path:
            anno_path = img_path.replace('Images', 'Annotations')
            suffix = Path(anno_path).suffix
            anno_path = anno_path.replace(suffix, '.txt')
            if 'Images' in img_path and os.path.exists(anno_path):
                return anno_path
        # LDTest数据集
        if 'LDTest' in img_path:
            anno_path = img_path.replace('Images', 'Annotations')
            suffix = Path(anno_path).suffix
            anno_path = anno_path.replace(suffix, '.txt')
            if 'Images' in img_path and os.path.exists(anno_path):
                return anno_path
       
         
        # raise NameError(img_path)
