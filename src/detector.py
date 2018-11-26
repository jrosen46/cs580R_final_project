#!/usr/bin/env python

"""
test_object_detection.py

This object detection uses the Tensorflow Object Detection API located at:
https://github.com/tensorflow/models/tree/master/research/object_detection

Some ideas were inspired by the following repositories:
    https://github.com/osrf/tensorflow_object_detector

"""
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import time

import numpy as np
import tensorflow as tf
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    err_msg = ('Please upgrade your TensorFlow installation '
               'to v1.9.* or later!')
    raise ImportError(err_msg)


# needed so that node can be run from anywhere
SRC_DIR = os.path.dirname(os.path.abspath(__file__))


class ObjectDetector(object):

    def __init__(self):

        rospy.init_node('detector', anonymous=False)

        # double check these topics ...
        rospy.Subscriber("image", Image, self.run_inference_for_single_image,
                         queue_size=1, buff_size=2**24)

        # tensorflow model
        self.detection_graph = self._init_detection_network()
        self.tensor_dict = self._init_tensor_handles()
        self.label_map = self._load_labels()
        self.sess = self._init_sess()

        # opencv interface
        self.bridge = CvBridge()

    def _download_and_extract_model(self):
        """Downloads and extracts object detection pretrained model.

        Returns
        -------
        frozen_graph_path : str
            Path to the frozen model.
        """
        # Model to download ... look at model zoo for other options.
        download_base = 'http://download.tensorflow.org/models/object_detection/'
        model_file = 'ssd_mobilenet_v1_coco_2017_11_17.tar.gz'
        model_path = os.path.join(SRC_DIR, 'models', model_file)

        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_path)
        tar_file = tarfile.open(model_path)
        for file_ in tar_file.getmembers():
            file_name = os.path.basename(file_.name)
            if file_name == 'frozen_inference_graph.pb':
                tar_file.extract(file_, os.path.join(SRC_DIR, 'models'))

        # Path to frozen detection graph. This is the actual model that is used for
        # the object detection.
        frozen_graph_path = os.path.join(SRC_DIR,
                                         'models',
                                         model_file.partition('.')[0],
                                         'frozen_inference_graph.pb')
        return frozen_graph_path
    
    def _load_frozen_graph(self, frozen_graph_path):
        """Loads frozen graph into memory.

        Parameters
        ----------
        frozen_graph_path : str
            Path to the frozen graph.

        Returns
        -------
        detection_graph : tf.Graph instance
            Represents the frozen graph with all pretrained weights.
        """
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _init_detection_network(self):
        """Initialize the object detection graph

        Returns
        -------
        detection_graph : tf.Graph instance
            Represents the frozen graph with all pretrained weights.
        """
        path = self._download_and_extract_model()
        detection_graph = self._load_frozen_graph(path)
        return detection_graph

    def _init_tensor_handles(self):
        """Initializes a handle to important tensors in graph.

        Returns
        -------

        """
        assert hasattr(self, 'detection_graph')

#        # Get handles to tensors
#        tensor_names = [
#            'num_detections', 'detection_boxes', 'detection_scores',
#            'detection_classes', 'detection_masks',
#        ]
#        tensor_dict = {
#            name: self.detection_graph.get_tensor_by_name(name + ':0')
#            for name in tensor_names
#        }
#
#        # The following processing is only for single image
#        boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
#        masks = tf.squeeze(tensor_dict['detection_masks'], [0])
#        # Reframe is required to translate mask from box coordinates to
#        # image coordinates and fit the image size.
#        num_detect = tf.cast(tensor_dict['num_detections'][0], tf.int32)
#        boxes = tf.slice(boxes, [0, 0], [num_detect, -1])
#        masks = tf.slice(masks, [0, 0, 0], [num_detect, -1, -1])
#        masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
#            masks, boxes, image_np.shape[0], image_np.shape[1])
#        masks_reframed = tf.cast(tf.greater(masks_reframed, 0.5), tf.uint8)
#        # Follow the convention by adding back the batch dimension
#        tensor_dict['detection_masks'] = tf.expand_dims(masks_reframed, 0)
#
#        return tensor_dict

        # Get handles to tensors
        tensor_names = [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes'
        ]
        tensor_dict = {
            name: self.detection_graph.get_tensor_by_name(name + ':0')
            for name in tensor_names
        }

        return tensor_dict

    def _load_labels(self):
        """Load the label map.

        Returns
        -------
        label_map :
        """
        # List of the strings that is used to add correct label for each box.
        path_to_labels = os.path.join(SRC_DIR, 'labels', 'mscoco_label_map.pbtxt')
        label_map = label_map_util.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)
        return label_map

    def _init_sess(self):
        """Initializes a tensorflow session.

        Returns
        -------
        tf.Session
        """
        assert hasattr(self, 'detection_graph')
        return tf.Session(graph=self.detection_graph)

    def convert_to_np_array(self):
        """
        """
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return np.asarray(image)

    def run_inference_for_single_image(self, image_np):
        """
        """
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = self.sess.run(
            self.tensor_dict,
            feed_dict={image_tensor: np.expand_dims(image_np, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = (output_dict['detection_classes'][0]
                                            .astype(np.uint8))
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        #output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict


if __name__ == '__main__':
    try:
        detector = ObjectDetector()
    except rospy.ROSInterruptException:
        pass
