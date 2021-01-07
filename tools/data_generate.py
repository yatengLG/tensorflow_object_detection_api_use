# -*- coding: utf-8 -*-
# @Author  : LG

import os
import imghdr
import xml.etree.ElementTree as ET
from lxml import etree
import tensorflow as tf
import cv2
import hashlib
from object_detection.utils import dataset_util, label_map_util
from tqdm import tqdm


class Generator(object):
    def __init__(self, imgs_path:str, xmls_path:str, train_file:str, eval_file:str):
        self.imgs_path = imgs_path
        self.xmls_path = xmls_path
        with open(train_file, 'r') as f:
            self.train_list = [line.rstrip('\n') for line in f.readlines()]
        with open(eval_file, 'r') as f:
            self.eval_list = [line.rstrip('\n') for line in f.readlines()]
        self.img_format = os.listdir(self.imgs_path)[0].split('.')[-1]

    def genetate(self):
        self.generate_label_pbtxt()
        self.label_map_dict = label_map_util.get_label_map_dict(os.path.join(os.getcwd(), 'datas', 'label.pbtxt'))
        self.generate_record()
        return True

    def generate_label_pbtxt(self):
        """
        依照所有的xml文件生成类别pdtxt
        :return:
        """
        xmls_list = [xml for xml in os.listdir(self.xmls_path) if xml.endswith('.xml')]
        data_label = set()

        for xml in xmls_list:
            tree = ET.parse(os.path.join(self.xmls_path, xml))
            for obj in tree.findall('object'):
                name = obj.find('name').text
                data_label.add(name)
        with open(os.path.join(os.getcwd(), 'datas', 'label.pbtxt'), 'w') as f:
            for i, name in enumerate(data_label):
                content = """item {\n  id:%d\n  name:'%s'\n}\n""" % (i+1, name)
                f.write(content)

    def generate_record(self):
        """
        生成tfrecord数据文件
        :return:
        """
        # 依次处理训练集和验证集
        for dataset in ['train', 'eval']:
            writer = tf.python_io.TFRecordWriter(
                os.path.join(os.getcwd(), 'datas', '{}.record'.format(dataset)))
            data_list = 'self.{}_list'.format(dataset)
            data_list = eval(data_list)
            pbar = tqdm(data_list)

            for img_name in pbar:
                pbar.set_description(dataset + ':')
                img_path = os.path.join(self.imgs_path, img_name+'.'+self.img_format)
                xml_path = os.path.join(self.xmls_path, img_name+'.xml')

                # 图片是否完好
                if not imghdr.what(img_path):
                    raise ValueError("img file {} error".format(img_path))

                with tf.gfile.GFile(img_path, 'rb') as f:
                    encoded_jpg = f.read()

                img = cv2.imread(img_path)
                height = img.shape[0]
                width = img.shape[1]
                key = hashlib.sha256(encoded_jpg).hexdigest()

                # 读取xml
                with tf.gfile.GFile(xml_path, 'rb') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                annotation = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                if 'object' not in annotation:
                    raise ValueError("xml file {} not have object".format(xml_path))

                if annotation is None:
                    raise ValueError("xml file {} error".format(xml_path))

                # object遍历
                xmin = []
                xmax = []
                ymin = []
                ymax = []
                classes_text = []
                classes = []
                poses = []
                truncated = []
                difficult = []
                for obj in annotation['object']:
                    xmin.append(float(obj['bndbox']['xmin']) / float(annotation['size']['width']))
                    ymin.append(float(obj['bndbox']['ymin']) / float(annotation['size']['height']))
                    xmax.append(float(obj['bndbox']['xmax']) / float(annotation['size']['width']))
                    ymax.append(float(obj['bndbox']['ymax']) / float(annotation['size']['height']))
                    classes_text.append(obj['name'].encode('utf8'))
                    classes.append(self.label_map_dict[obj['name']])
                    poses.append(obj['pose'].encode('utf8'))
                    truncated.append(int(obj['truncated']))
                    difficult.append(int(bool(int(obj['difficult']))))

                # feature dict
                feature_dict = {
                    'image/height': dataset_util.int64_feature(height),
                    'image/width': dataset_util.int64_feature(width),
                    'image/filename': dataset_util.bytes_feature(annotation['filename'].encode('utf8')),
                    'image/source_id': dataset_util.bytes_feature(annotation['filename'].encode('utf8')),
                    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                    'image/format': dataset_util.bytes_feature(self.img_format.encode('utf8')),
                    'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
                    'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
                    'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
                    'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
                    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                    'image/object/class/label': dataset_util.int64_list_feature(classes),
                    'image/object/difficult': dataset_util.int64_list_feature(difficult),
                    'image/object/truncated': dataset_util.int64_list_feature(truncated),
                    'image/object/view': dataset_util.bytes_list_feature(poses),
                }

                # 写入tfrecord
                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                writer.write(example.SerializeToString())
            writer.close()
