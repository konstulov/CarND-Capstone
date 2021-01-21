from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import tensorflow as tf

SSD_GRAPH_FILE = '/home/workspace/data/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.prev_log_time = rospy.get_time() - 1
        self.detection_graph = self.load_graph(SSD_GRAPH_FILE)
        
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        
        self.sess = tf.Session(graph=self.detection_graph)

    @staticmethod
    def load_graph(graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    @staticmethod
    def filter_boxes(min_score, boxes, scores, classes, cls=None):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score and (cls is None or classes[i] == cls):
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def pipeline_img(self, img, min_conf=0.8, cls=None):
        boxes, scores, classes = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                               feed_dict={self.image_tensor: np.expand_dims(img, 0)})
        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # Filter boxes with a confidence score less than min_conf
        boxes, scores, classes = self.filter_boxes(min_conf, boxes, scores, classes, cls)

        return scores, classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        img = np.asarray(image)
        # cls==10 is for traffic light
        scrores, classes = self.pipeline_img(img, min_conf=0.5, cls=10)
        if rospy.get_time() - self.prev_log_time >= 1:
            self.prev_log_time = rospy.get_time()
            rospy.logwarn('TLClassifier.get_classification(): len(scrores) = %d' % len(scrores))
        return TrafficLight.UNKNOWN
