#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : freeze_graph.py
#   Author      : Vybt
#   Created date: 2019-09-30 15:57:33
#   Description :
#
#================================================================


import os
import tensorflow as tf
from core.yolov3 import YOLOV3


tf.app.flags.DEFINE_integer('model_version', 1, 'Models version number.')
tf.app.flags.DEFINE_string('work_dir', './checkpoint', 'Working directory.')
tf.app.flags.DEFINE_string('model_model', "YOLOv3", 'Model id name to be loaded.')
tf.app.flags.DEFINE_string('export_model_dir', "./checkpoint/versions", 'Directory where the model exported files should be placed.')

ckpt_file = "./checkpoint/yolov3_person.ckpt-50"
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

FLAGS = tf.app.flags.FLAGS
FLAGS = tf.app.flags.FLAGS
model_name = FLAGS.model_model
log_folder = FLAGS.work_dir

def main(_):
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')

    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[], dtype=tf.float32), }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)

    tf_example['x'] = tf.reshape(tf_example['x'], (1, 416, 416, 3))
    input_tensor = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
    
    model = YOLOV3(input_data, trainable=False)
    print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    # Create SavedModelBuilder class
    # defines where the model will be exported
    export_path_base = FLAGS.export_model_dir
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_input = tf.saved_model.utils.build_tensor_info(input_tensor)
    tensor_conv_sbbox_output = tf.saved_model.utils.build_tensor_info(model.conv_sbbox)
    tensor_conv_mbbox_output = tf.saved_model.utils.build_tensor_info(model.conv_mbbox)
    tensor_conv_lbbox_output = tf.saved_model.utils.build_tensor_info(model.conv_lbbox)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_input},
            outputs={
                'conv_sbbox': tensor_conv_sbbox_output,
                'conv_mbbox': tensor_conv_mbbox_output,
                'conv_lbbox': tensor_conv_lbbox_output
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
        })

    # export the model
    builder.save(as_text=True)
    print('Done exporting!')

if __name__ == '__main__':
    tf.app.run()
