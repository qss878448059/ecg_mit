import os
import cv2
import numpy as np
import random
import json
_split_percentage = .70
_split_validation_percentage = 0.70
_split_test_percentage = 0.50
dir_data = 'Data/dataset_filtered/'
_size = (128, 128)
_n_classes = 8

labels_json = '{ ".": "NOR", "N": "NOR", "V": "PVC", "/": "PAB", "L": "LBB", "R": "RBB", "A": "APC", "!": "VFW", "E": "VEB" }'
labels_to_float = '{ "NOR": "0", "PVC" : "1", "PAB": "2", "LBB": "3", "RBB": "4", "APC": "5", "VFW": "6", "VEB": "7" }'
float_to_labels = '{ "0": "NOR", "1" : "PVC", "2": "PAB", "3": "LBB", "4": "RBB", "5": "APC", "6": "VFW", "7": "VEB" }'
labels = json.loads(labels_to_float)
revert_labels = json.loads(float_to_labels)
original_labels = json.loads(labels_json)

# 加载数据名称，返回训练集、验证集的图片名称
def load_files(directory):
    train = []
    validation = []
    # test = []

    classes = {'NOR', 'PVC', 'PAB', 'LBB', 'RBB', 'APC', 'VFW', 'VEB'}

    classes_dict = dict()

    for key in classes:
        classes_dict[key] = [f for f in os.listdir(directory) if key in f if f[-5] == '0']
        random.shuffle(classes_dict[key])

    for _, item in classes_dict.items():
        train += item[: int(len(item) * _split_validation_percentage)]
        val = item[int(len(item) * _split_validation_percentage):]
        validation += val[: int(len(val) * _split_test_percentage)]
        # test += val[int(len(val) * _split_test_percentage):]

    random.shuffle(train)
    random.shuffle(validation)
    return train, validation


# 返回图片标签
def encode_label(file):
    label = [0 for _ in range(_n_classes)]
    label[int(labels[file[:3]])] = 1
    return label


# 返回图片集和标签集
def pre_data(dir, all_img, size, encode_labels=True):
    images = []
    labes = []
    # all_img = os.listdir(dir)
    for img_name in all_img:
        img = cv2.imread(dir + img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
        img = img.astype('float32')
        img /= 255
        img = img * 2.0 - 1.0
        img = np.reshape(img, (1, 128, 128))
        if encode_labels:
            label = encode_label(img_name)
        else:
            label = labels[img_name[:3]]
        images.append(img)
        labes.append(label)
    x = np.array(images)
    y = np.array(labes)
    return x, y


def load_data(dir, all_img, size, batch_size):
    img, labels = pre_data(dir, all_img, size)

    def reader():
        batch_start = 0
        batch_end = batch_size
        while batch_start + batch_size < len(img):
            batch_img = []
            batch_labes = []
            for i in range(batch_start, batch_end):
                batch_img.append(img[i])
                batch_labes.append(labels[i])
            batch_img_arr = np.array(batch_img)
            batch_labes_arr = np.array(batch_labes).astype('float32')
            batch_labes_arr = batch_labes_arr.argmax(1).astype('int64')
            yield batch_img_arr, batch_labes_arr
            batch_start = batch_start + batch_size
            batch_end = batch_end + batch_size
        # print('打包。。。')
        if batch_start < len(img):
            batch_img = []
            batch_labes = []
            for i in range(batch_start, len(img)):
                batch_img.append(img[i])
                batch_labes.append(labels[i])
            batch_img_arr = np.array(batch_img)
            batch_labes_arr = np.array(batch_labes).astype('float32')
            batch_labes_arr = batch_labes_arr.argmax(1).astype('int64')
            yield batch_img_arr, batch_labes_arr

    return reader