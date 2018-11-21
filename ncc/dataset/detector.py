from glob import glob
import numpy as np
import os
import pandas as pd
import pickle
import xml.etree.ElementTree as ET


def search_class_names(xml_dir):
    """ Search class names from xml files.
    """
    class_names = []
    if not os.path.basename(xml_dir):
        xml_dir = os.path.join(xml_dir, '*.xml')
    for xml in glob(xml_dir):
        tree = ET.parse(xml)
        root = tree.getroot()
        for object_tree in root.findall('object'):
            class_name = object_tree.find('name').text
            class_names.append(class_name)

    return sorted(list(set(class_names)))


def make_class_csv(xml_dir=None, class_names=None, save_path='./class.csv'):
    """ Make class csv.
    """
    if xml_dir and not class_names:
        class_names = search_class_names(xml_dir)
    with open(save_path, 'w') as csv:
        for class_name in class_names:
            csv.write(class_name + ',' + str(class_names.index(class_name)) + '\n')


def make_target_csv(xml_dir, img_dir, save_path='./target.csv'):
    """ Make target csv.
    """
    if not os.path.basename(xml_dir):
        xml_dir = os.path.join(xml_dir, '*.xml')
    if os.path.basename(img_dir):
        img_dir = os.path.dirname(os.path.abspath(img_dir))
    with open(save_path, 'w') as csv:
        for xml in sorted(glob(xml_dir)):
            tree = ET.parse(xml)
            root = tree.getroot()
            image = root.find('filename').text
            image = os.path.basename(image)
            image_path = os.path.join(img_dir, image)
            if not os.path.exists(image_path):
                print(image_path + 'does not exists.')
                continue
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                for bndbox in object_tree.iter('bndbox'):
                    xmin = str(bndbox.find('xmin').text)
                    ymin = str(bndbox.find('ymin').text)
                    xmax = str(bndbox.find('xmax').text)
                    ymax = str(bndbox.find('ymax').text)
                target = [image_path, xmin, ymin, xmax, ymax, class_name]
                csv.write(','.join([str(t) for t in target]) + '\n')


def clip_box(xml_dir):
    """ Clip bounding box coords into range of the image size.
    """
    if not os.path.basename(xml_dir):
        xml_dir = os.path.join(xml_dir, '*.xml')
    for xml in glob(xml_dir):
        tree = ET.parse(xml)
        root = tree.getroot()
        size_tree = root.find('size')
        width = size_tree.find('width').text
        height = size_tree.find('height').text
        for object_tree in root.findall('object'):
            for bndbox in object_tree.iter('bndbox'):
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                if xmin < 0:
                    bndbox.find('xmin').text = 0
                    tree.write(xml)
                    print('xmin < 0, clip ' + str(xml))
                if ymin < 0:
                    bndbox.find('ymin').text = 0
                    tree.write(xml)
                    print('ymin < 0, clip ' + str(xml))
                if xmax > float(width):
                    bndbox.find('xmax').text = width
                    tree.write(xml)
                    print('xman > width, clip ' + str(xml))
                if ymax > float(height):
                    bndbox.find('ymax').text = height
                    tree.write(xml)
                    print('ymax > height, clip ' + str(xml))