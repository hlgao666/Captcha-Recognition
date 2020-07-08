import csv
from PIL import Image
import os
import numpy as np


def convert_img_to_csv(img_dir, csv_save_path, mode):
    # 设置需要保存的csv路径
    if mode == 0:   # 训练模式
        with open(f'./{csv_save_path}.csv', 'w', newline='') as f:
            # 设置csv文件的列名
            column_name = ['label']
            column_name.extend([f'pixel{i}' for i in range(16 * 20)])
            writer = csv.writer(f)
            writer.writerow(column_name)
            # 该目录下有10个目录,目录名从0-9
            for i in range(10):
                # 获取目录的路径
                img_temp_dir = os.path.join(img_dir, str(i))
                img_list = os.listdir(img_temp_dir)
                for img_name in img_list:
                    # 判断文件是否为目录,如果为目录则不处理
                    if not os.path.isdir(img_name):
                        img_path = os.path.join(img_temp_dir, img_name)
                        img = Image.open(img_path)
                        img = np.array(img, 'f').flatten()
                        row_data = [i]
                        row_data.extend(np.trunc(img).astype(int).tolist())
                        writer.writerow(row_data)

    else:   # 测试模式 img_dir: 'D:/Test_dataset'; save_path: 'D:/test'
        with open(f'{csv_save_path}.csv', 'w', newline='') as f:
            # 设置csv文件的列名
            column_name = ([f'pixel{i}' for i in range(16 * 20)])
            writer = csv.writer(f)
            writer.writerow(column_name)

            img_list = os.listdir(img_dir)
            # 对文件名排序
            img_list.sort(key=lambda x: int(x[:-4]))
            for img_name in img_list:
                # 判断文件是否为目录,如果为目录则不处理
                if not os.path.isdir(img_name):
                    img_path = os.path.join(img_dir, img_name)
                    img = Image.open(img_path)
                    img = np.array(img, 'f').flatten()
                    row_data = (np.trunc(img).astype(int).tolist())
                    writer.writerow(row_data)
