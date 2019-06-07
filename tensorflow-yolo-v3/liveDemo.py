# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time
import cv2
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image

IM_WIDTH = 640
IM_HEIGHT = 480
camera = cv2.VideoCapture(0)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', '', 'Input image')
tf.app.flags.DEFINE_string(
    'output_img', 'output.png', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', '', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', True, 'Use tiny version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.0, 'Gpu memory fraction to use')

def main(argv=None):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )
    model = yolo_v3_tiny.yolo_v3_tiny
    classes = load_coco_names(FLAGS.class_names)

    while True:
        tf.reset_default_graph()
        __, img = camera.read()
        img = Image.fromarray(img)
        img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)
        img_resized = img_resized.astype(np.float32)

        boxes, inputs = get_boxes_and_inputs(model, len(classes), FLAGS.size, FLAGS.data_format)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        with tf.Session(config=config) as sess:
            saver.restore(sess, FLAGS.ckpt_file)
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})

        filtered_boxes = non_max_suppression(detected_boxes,
                                             confidence_threshold=FLAGS.conf_threshold,
                                             iou_threshold=FLAGS.iou_threshold)

        draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), True)

        #print(filtered_boxes)
        #img.save(FLAGS.output_img)
        # show the output images
        opencvimage = np.array(img)
        cv2.imshow("Video Feed", opencvimage)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    tf.app.run()
