# -*- coding: utf-8 -*-
# @Author  : LG


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from object_detection import model_hparams
from object_detection import model_lib
from object_detection.utils import config_util
import os
import shutil
from tools.data_generate import Generator
import wget
import tarfile
import argparse
from google.protobuf import text_format


tf.logging.set_verbosity(tf.logging.INFO)


with open(os.path.join(os.getcwd(), 'pretrained_models/model_zoo'), 'r') as f:
    model_dict = {line.rstrip('\n').split(' ')[0]: line.rstrip('\n').split(' ')[1] for line in f.readlines()}

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="ssd_mobilenet_v1_coco", type=str, help="Pretrained_model.")
parser.add_argument("--pipeline_config", default=None, type=str, help="Model pipeline config file.")
parser.add_argument("--train_record", default=None, type=str, help="Train Data record.")
parser.add_argument("--eval_record", default=None, type=str, help="Eval Data record.")
parser.add_argument("--label_pbtxt", default=None, type=str, help="Train and Eval data label file")
parser.add_argument("--images_path", default=None, type=str, help="Train and Eval images path, be used to generate tfrecord.")
parser.add_argument("--xmls_path", default=None, type=str, help="Train and Eval xmls path, be used to generate tfrecord.")
parser.add_argument("--train_txt", default=None, type=str, help="Train image name file, be used to generate tfrecord.")
parser.add_argument("--eval_txt", default=None, type=str, help="Eval image name file, be used to generate tfrecord.")

args = parser.parse_args()

# 模型准备
model_name = args.model_name
train_record = args.train_record
eval_record = args.eval_record
label_pbtxt = args.label_pbtxt
pipeline_config = args.pipeline_config
images_path = args.images_path
xmls_path = args.xmls_path
train_txt = args.train_txt
eval_txt = args.eval_txt

if model_name is None:
    raise ValueError("model_name are required.")

# 模型准备
if model_name not in model_dict.keys():
    raise ValueError('only support models:\n  {}'.format('\n  '.join(model_dict.keys())))

pretrained_weight_tar = os.path.join(os.getcwd(), 'pretrained_models', model_dict[model_name].split('/')[-1])
if not os.path.exists(pretrained_weight_tar):
    try:
        tf.logging.info('Downloading {} pretrained weight from {}.'.format(model_name, model_dict[model_name]))
        wget.download(model_dict[model_name], os.path.join(os.getcwd(), 'pretrained_models'))
    except tf.errors.UnavailableError:
        tf.logging.error('Download failed!')

try:
    t = tarfile.open(pretrained_weight_tar)
    t.extractall(path=os.path.join(os.getcwd(), 'pretrained_models'))

except tf.errors.UnavailableError:
    tf.logging.error('untar {} failed!'.format(pretrained_weight_tar))

# 数据准备
if not (train_record and eval_record and label_pbtxt):
    if images_path and xmls_path and train_txt and eval_txt:
        tf.logging.info("Generate TFrecord from {} and {}".format(images_path, xmls_path))
        generator = Generator(images_path, xmls_path, train_txt, eval_txt)
        generator.genetate()
        train_record = os.path.join(os.getcwd(), 'datas', 'train.record')
        eval_record = os.path.join(os.getcwd(), 'datas', 'eval.record')
        label_pbtxt = os.path.join(os.getcwd(), 'datas', 'label.pbtxt')
    else:
        raise ValueError("For generate data record, images_path and xmls_path and train_txt and eval_txt must be required!")

# config 准备
if pipeline_config is None:
    model_name = model_dict[model_name].split('/')[-1].rstrip('.tar.gz')
    pipeline_config = os.path.join(os.getcwd(), 'configs', '{}.config'.format(model_name))
    tf.logging.info("Copy config file to {}".format(pipeline_config))
    shutil.copy(os.path.join(os.getcwd(), 'pretrained_models', model_name, 'pipeline.config'), pipeline_config)
    config = config_util.get_configs_from_pipeline_file(pipeline_config)

    with open(label_pbtxt, 'r') as f:
        num_classes = ''.join(f.readlines()).count("item {")

    config_updata = {
        'model.{}.num_classes'.format(config['model'].WhichOneof('model')): num_classes,
        'train_config.fine_tune_checkpoint': os.path.join(os.getcwd(), 'pretrained_models', model_name, 'model.ckpt'),
        'label_map_path': label_pbtxt,
        'train_input_path': train_record,
        'eval_input_path': eval_record,
    }
    # 更新config文件
    tf.logging.info("Updata config file {}".format(pipeline_config))
    config = config_util.merge_external_params_with_configs(config, kwargs_dict=config_updata)
    config = config_util.create_pipeline_proto_from_configs(config)
    with tf.gfile.Open(pipeline_config, "wb") as f:
        f.write(text_format.MessageToString(config))

# 训练结果保存位置
save_path = os.path.join(os.getcwd(), 'weights', model_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)


def main(unused_argv):
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=tf.estimator.RunConfig(model_dir=save_path),
        hparams=model_hparams.create_hparams(None),
        pipeline_config_path=pipeline_config,
        train_steps=None,
        sample_1_of_n_eval_examples=1,
        sample_1_of_n_eval_on_train_examples=(5))
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


if __name__ == '__main__':
  tf.app.run()